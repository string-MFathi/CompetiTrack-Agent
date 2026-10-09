import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated

# استيراد أنواع الرسائل لتمييز توجيهات النظام (System) عن مدخلات المستخدم (Human)
from langchain_core.messages import SystemMessage, HumanMessage
# استيراد فئة الاتصال بنموذج Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
# استيراد أداة البحث المحدثة من Tavily
from langchain_tavily import TavilySearch
# استيراد أدوات بناء مسار العمل (Graph) ونقطة النهاية (END)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# استيراد عقدة جاهزة لتنفيذ الأدوات بدلاً من برمجتها من الصفر
from langgraph.prebuilt import ToolNode
# استيراد نوع الخطأ الخاص بتجاوز الحد الأقصى للبحث لاصطياده في استراتيجية التراجع
from langgraph.errors import GraphRecursionError

# 1. تحميل متغيرات البيئة 
# هذه الدالة تقرأ ملف .env وتسحب منه المفاتيح (Google, Tavily, LangSmith) لتفعيلها في المشروع
load_dotenv()

# 2. تصميم الذاكرة المركزية (State)
# State هي الذاكرة التي تنتقل بين جميع العُقد. هنا نحدد أنها ستحتوي على قائمة من "الرسائل"،
# الدالة add_messages تضمن إضافة الرسائل الجديدة للقديمة بدلاً من مسحها (Append vs Overwrite).
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 3. إعداد أداة البحث ونموذج جيميناي 
# تحديد أداة البحث والسماح لها بجلب 3 نتائج كحد أقصى في كل عملية بحث لتوفير الرصيد
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# استدعاء أحدث إصدار من نموذج جيميناي، وضبط درجة الحرارة (temperature=0) لجعله دقيقاً وغير خيالي
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
# ربط الأدوات بالنموذج ليصبح قادراً على اتخاذ قرار باستدعاء أداة البحث عندما يحتاج بيانات خارجية
llm_with_tools = llm.bind_tools(tools)

# 4. توجيه النموذج (System Prompt)
# هذا هو "عقل" الوكيل والقواعد التي تحكم عمله. مكتوب بالإنجليزية لضمان دقة التنفيذ والمخرجات.
# أضفنا له تعليمات التوقف (Guardrails) إذا لم يجد بيانات بعد 3-4 محاولات لمنع الحلقة المفرغة.
system_prompt = SystemMessage(content="""
You are a Competitive Intelligence and Market Research Analyst. 
Your task is to conduct comprehensive competitive research on the requested company using the provided search tool.
Extract 4 main pillars accurately:
1. Pricing model and subscription tiers.
2. Core competitive advantages and key features.
3. Target audience and product weaknesses/drawbacks based on customer reviews.
4. Market gaps and opportunities.
Keep searching until you find clear facts. HOWEVER, if you perform 3 to 4 searches and cannot find specific public data (e.g., pricing or reviews), DO NOT get stuck in a loop. Stop searching, clearly state in your report that this specific data is currently unavailable publicly, and compile the final report using only the data you have gathered so far. Do not invent any information.
Finally, write a structured, analytical final report strictly in English.
""")

# 5. تصميم العُقد (Nodes)
# العقدة الأولى: العقل المفكر (الوكيل). تقرأ الذاكرة، تضيف التوجيه (Prompt) إذا لم يكن موجوداً، ثم تفكر.
def researcher_node(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_prompt] + messages
        
    # استدعاء النموذج للرد على الرسائل الحالية
    response = llm_with_tools.invoke(messages)
    # إرجاع الرد ليتم إضافته تلقائياً إلى الذاكرة (AgentState)
    return {"messages": [response]} 

# العقدة الثانية: المنفذ. عقدة جاهزة تستلم طلب البحث من الوكيل، تنفذه عبر Tavily، وتعيد النتائج.
tool_executor_node = ToolNode(tools)

# 6. صانع القرار (Conditional Edge)
# هذه الدالة تمثل "شرط المرور". تنظر لآخر رسالة أخرجها الوكيل:
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # إذا كانت الرسالة تحتوي على طلب استخدام أداة (tool_calls)، نوجه المسار لعقدة التنفيذ.
    if last_message.tool_calls:
        return "continue"
    # إذا قرر الوكيل أنه انتهى وكتب التقرير، ننهي مسار العمل.
    return "end"

# 7. هندسة مسار العمل (Workflow Graph)
workflow = StateGraph(AgentState)

# إضافة العُقد (نقاط التوقف) إلى الرسم البياني
workflow.add_node("agent", researcher_node)
workflow.add_node("action", tool_executor_node)

# تحديد نقطة البداية الدائمة وهي عقدة التفكير (الوكيل)
workflow.set_entry_point("agent")

# ربط العُقد ببعضها باستخدام صانع القرار
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action", # إذا أراد البحث، اذهب لعقدة action
        "end": END            # إذا انتهى، اخرج من البرنامج
    }
)
# بعد انتهاء عقدة البحث (action)، يجب أن يعود المسار إجبارياً للوكيل ليقرأ النتائج
workflow.add_edge("action", "agent")

# تجميع (Compile) الرسم البياني وتحويله لبرنامج قابل للتشغيل
competitor_agent = workflow.compile()

# 8. حلقة التشغيل والمراقبة
if __name__ == "__main__":
    # طباعة واجهة المستخدم باللغة الإنجليزية لتجنب مشاكل تشفير الحروف في الـ Terminal
    print("🤖 Welcome to the Competitive Intelligence Agent [CompetiTrack-Agent]\n")
    print("-" * 50)
    
    # استقبال اسم الشركة ورابطها من المستخدم بشكل تفاعلي
    competitor_name = input("📌 Please enter the competitor's company name: ")
    competitor_url = input("🔗 Please enter the competitor's official URL: ")
    
    print("\n⏳ Initiating research and analysis... Please wait.\n")
    
    # دمج المدخلات في سياق يسهل على الوكيل فهمه
    user_input = f"Conduct a comprehensive competitive research on the following company/platform: Competitor Name: {competitor_name} | Official URL: {competitor_url}"
    
    # تهيئة الذاكرة المبدئية برسالة المستخدم
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    # متغير مساعد نحفظ فيه الذاكرة خطوة بخطوة، لكي لا نفقدها إذا حدث خطأ مفاجئ
    current_state = initial_state 
    
    # استخدام try/except لتطبيق "استراتيجية التراجع" (Fallback) عند فشل الوكيل في التوقف
    try:
        # تشغيل الوكيل مع وضع حد أقصى (5 دورات) لحمايتك من استهلاك الرصيد
        for event in competitor_agent.stream(initial_state, stream_mode="values", config={"recursion_limit": 5}):
            current_state = event # تحديث الذاكرة الاحتياطية في كل خطوة
            last_msg = event["messages"][-1]
            
            # طباعة ما يحدث في الكواليس بشفافية للمستخدم
            if last_msg.type == "ai" and last_msg.tool_calls:
                print(f"🔍 Agent decided to search for: {last_msg.tool_calls[0]['args']['query']}")
            elif last_msg.type == "tool":
                print("✅ Data fetched from the internet and added to memory.")
                print("-" * 50)
            elif last_msg.type == "ai" and not last_msg.tool_calls:
                print("📝 Final Report is ready:\n")
                print(last_msg.content)
                
    except GraphRecursionError:
        # إذا تخطى الوكيل 5 دورات، سيصطاد البرنامج الخطأ هنا بدلاً من الانهيار
        print("\n⚠️ Alert: Search limit reached (5 attempts).")
        print("💡 Fallback Strategy: Forcing the agent to write a report using the partially gathered data...\n")
        
        # سحب كل البيانات التي تم جمعها في المحاولات الخمس من الذاكرة الاحتياطية
        saved_messages = current_state["messages"]
        
        # إنشاء رسالة جديدة تُجبر الوكيل على التوقف عن البحث وكتابة التقرير بالبيانات الحالية فقط
        fallback_prompt = HumanMessage(content="Search limit reached. Stop searching immediately. Write the final structured report based ONLY on the data you found so far in the memory. If any of the 4 pillars lack data, clearly state 'Public data not available for this section'.")
        saved_messages.append(fallback_prompt)
        
        # استدعاء نموذج جيميناي مباشرة (بدون أدوات البحث) ليكتب التقرير النهائي وينقذ الموقف
        fallback_response = llm.invoke(saved_messages)
        
        print("📝 Partial Final Report (Best Effort):\n")
# التأكد من استخراج النص الصافي فقط سواء كان الرد سلسلة نصية (String) أو قائمة (List)
        final_text = fallback_response.content[0]["text"] if isinstance(fallback_response.content, list) else fallback_response.content
        print(final_text)