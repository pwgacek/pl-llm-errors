from openai import OpenAI
import os


api_key = os.environ["GROQ_API_KEY"]

sentence = """
The following paragraphs each describe a set of seven objects arranged in a fixed order. The statements are logically consistent within each paragraph. In a golf tournament, there were seven golfers: Ana, Eve, Ada, Dan, Rob, Amy, and Joe. Dan finished third. Ana finished above Ada. Amy finished last. Dan finished below Rob. Eve finished below Ada. Rob finished below Joe.

Options:
(A) Ana finished third
(B) Eve finished third
(C) Ada finished third
(D) Dan finished third
(E) Rob finished third
(F) Amy finished third
(G) Joe finished third
"""

prompt = f"""
Translate the following reasoning task into Polish.

Requirements:
- Preserve the exact logical meaning.
- Keep the structure of the task unchanged.
- Do not simplify the problem.

Return only the translated text, without explanations.

Text:
{sentence}
"""

print(prompt)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": prompt}
        ],
    temperature=0,
)

print(resp.choices[0].message.content)