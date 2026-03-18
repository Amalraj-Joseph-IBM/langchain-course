from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def main():
    print("Hello from langchain-course!")
    information = """
    Mohanlal Viswanathan (IPA: [moːhɐnlaːl ʋiʃʋʷɐn̪aːt̪ʰɐn]; born 21 May 1960), known mononymously as Mohanlal, is an Indian actor and filmmaker who predominantly works in Malayalam cinema and has also occasionally appeared in Tamil, Hindi, Telugu and Kannada films.[4][5][6] Mohanlal has a prolific career spanning over four decades, during which he has acted in more than 400 films.[5][7] The Government of India honoured him with Padma Shri in 2001[8] and Padma Bhushan in 2019, India's fourth and third highest civilian honours,[9] for his contributions to Indian cinema. In 2009, he became the first actor in India to be awarded the honorary rank of lieutenant colonel in the Territorial Army.[10][11] Mohanlal was named as one of "the men who changed the face of the Indian Cinema" by CNN.[12] In 2025, the Government of India honoured him with the Dadasaheb Phalke Award, the highest award in the field of Indian cinema, for his "outstanding contribution to the growth and development of Indian cinema."[13][14]

Mohanlal made his acting debut at age 18 in the Malayalam film Thiranottam in 1978, but the film was delayed in its release for 25 years due to censorship issues. His screen debut was in the 1980 romance film Manjil Virinja Pookkal, in which he played the antagonist.[15][16] He continued to do villainous roles and rose to secondary lead roles in the following years. By the mid-1980s, he established himself as a bankable leading actor and attained stardom after starring in several successful films in 1986; the crime drama Rajavinte Makan released that year heightened his stardom.[15] Mohanlal prefers to work in Malayalam films, but he has also appeared in other language films. Some of his best known non-Malayalam films include the Tamil political drama Iruvar (1997), the Hindi crime drama Company (2002) and the Telugu film Janatha Garage (2016).[17][18]

Mohanlal has won five National Film Awards—two Best Actor, a Special Jury Mention and a Special Jury Award for acting, and an award for Best Feature Film (as producer), also nine Kerala State Film Awards and Filmfare Awards South and numerous other accolades. He received honorary doctorates from Sree Sankaracharya University of Sanskrit in 2010[19] and the University of Calicut in 2018.[20]

Mohanlal is also known for his philanthropic endeavours. He founded the ViswaSanthi Foundation, a non-profit charitable organisation, to create and deliver high-impact and focused programmes to the underprivileged sections of society in the areas of healthcare and education.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatOpenAI(temperature=0, model="gpt-5")
    #llm = ChatOllama(temperature=0, model="gemma3:270m")
    llm = ChatOllama(temperature=0, model="gpt-oss")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
