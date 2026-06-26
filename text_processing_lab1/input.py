import tiktoken
from transformers import AutoTokenizer

RONALDO_PARAGRAPH = '''
Cristiano Ronaldo dos Santos Aveiro[a] (born 5 February 1985), nicknamed CR7, is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al-Nassr and the Portugal national team. Widely regarded as one of the greatest players in history and the greatest Portuguese player ever, he has won numerous individual accolades throughout his career, including five Ballons D'or, a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes. He was also named the world's best player five times by FIFA.[note 3]

Ronaldo is one of the most decorated players in the history of professional football, having won 35 trophies in his career, including five UEFA Champions Leagues, two UEFA Nations Leagues and the UEFA European Championship. He holds the records for most goals (140) and assists (42) in the Champions League, goals (14) and assists (8) in the European Championship, and most international appearances (227), men's international goals (143) and most international victories (138). He is the only player to have scored 100 goals with four different clubs and to finish as top scorer in four different domestic leagues. He has made over 1,300 professional career appearances, the most by an outfield player, and has scored over 970 official senior career goals for club and country, making him the top goalscorer of all time.

Born in Funchal, Madeira, Ronaldo began his career with Sporting CP before signing with Manchester United in 2003. He gradually established himself as an integral player for the club, and won three consecutive Premier League titles, the Champions League, and the FIFA Club World Cup. This resulted in the 2007–08 season, Ronaldo winning his second Premier League Player of the Season award and his first Ballon d'Or at age 23, making him the second Premier League player ever to win the award. In 2009, Ronaldo became the subject of the then-most expensive transfer in history when he joined Real Madrid in a deal worth €94 million (£80 million). At Madrid, he was at the forefront of the club's resurgence as a dominant European force, helping them win four Champions Leagues between 2014 and 2018, including the long-awaited La Décima in 2014, where he set the record for most goals scored in a Champions League season. He also won two La Liga titles, and became the club's all-time top goalscorer. This led him to win four Ballons D'or in 2013, 2014, 2016 and 2017.

Following issues with the club hierarchy, Ronaldo signed for Juventus in 2018 in a transfer worth a league record of €100 million, where he was pivotal in winning two consecutive Serie A titles and a Coppa Italia. He also won the Capocannoniere in 2021 as the league's top scorer and was named Serie A Footballer of the Year back-to-back in 2019 and 2020. In 2021, he returned to Manchester United, but had his contract terminated in 2022 after a dispute with the club's management. Ronaldo joined Al-Nassr in 2023, and led them to the Saudi Pro League title in 2026, while also finishing as the league top scorer back-to-back in 2024 and 2025.

Ronaldo made his international debut for Portugal in 2003 at the age of 18 and has earned more than 200 caps, making him history's most-capped male player.[7] He has played in eleven major tournaments. He scored his first international goal in Euro 2004, where he helped Portugal reach the final and subsequently made the team of the tournament. He assumed captaincy of the national team ahead of Euro 2008; and at Euro 2012, he was named in the team of the tournament. Ronaldo led Portugal to their first major tournament title at Euro 2016, being named in the team of the tournament for the third time. In the 2018 World Cup, he had his most prolific World Cup campaign with four goals. He received the Golden Boot as the top scorer of Euro 2020 before playing in his fifth World Cup at the 2022 World Cup. He has won two UEFA Nations Leagues, in 2019 and 2025, receiving the top scorer award in both finals.

One of the world's most marketable and famous athletes, Ronaldo has been ranked by Sportico as the third highest-paid athlete of all time in April 2026,[8][9] and as the highest-paid athlete of the year for three consecutive years between 2023 and 2025.[10][11][12][13][14] He was also ranked the world's highest-paid athlete by Forbes on five occasions, the most in history, and the world's most famous athlete by ESPN from 2016 to 2019. In 2026, Ronaldo appeared on the World’s Billionaires list for the first time at $1.2B net worth.[15] Time included him on their list of the 100 most influential people in the world in 2014. He is the most popular sportsperson on social media: he counts over 1 billion total followers across Facebook, Twitter, YouTube and Instagram, making him the first person to achieve that feat. Ronaldo was named in the UEFA Ultimate Team of the Year in 2015, the All-time UEFA Euro XI in 2016, and the Ballon d'Or Dream Team in 2020. In recognition of his record-breaking goalscoring success, he received special awards for Outstanding Career Achievement by FIFA in 2021 and Champions League All-Time Top Scorer by UEFA in 2024.
'''

PRICING = {
    "GPT-4o": {"input": 2.50,  "output": 10.00},
    "Claude Sonnet": {"input": 3.00,  "output": 15.00},
    "LLaMA-3 (self-hosted)": {"input": 0.0, "output": 0.0},  # your infra cost
}

TEST_CASES = {
    "english_prose": """Cristiano Ronaldo dos Santos Aveiro, born 5 February 1985,
    is a Portuguese professional footballer widely regarded as one of the greatest
    players in history. He has won five Ballon d'Or awards and holds the record
    for most international goals with 143.""",

    "python_code": """
def bm25_score(query_terms, doc_terms, doc_freq, num_docs, avg_doc_len, k1=1.5, b=0.75):
    score = 0.0
    doc_len = len(doc_terms)
    for term in query_terms:
        if term not in doc_freq:
            continue
        tf = doc_terms.count(term)
        df = doc_freq[term]
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        score += idf * tf_norm
    return score
""",

    "tamil": "கிறிஸ்டியானோ ரொனால்டோ உலகின் மிகச்சிறந்த கால்பந்து வீரர்களில் ஒருவராக கருதப்படுகிறார். அவர் ஐந்து பலோன் டி'ஓர் விருதுகளை வென்றுள்ளார்.",

    "japanese": "クリスティアーノ・ロナウドは、史上最も偉大なサッカー選手の一人と広く見なされています。彼は5回バロンドールを受賞しています。",

    "emojis": "Ronaldo scores again! 🔥⚽🏆🇵🇹 What a player! 🐐💪🎯✨🙌 The greatest of all time! 🥇🌟💫🎉🏅",

    "mixed_noise": "€94M £80M 2003–2026 #CR7 @Ronaldo $1.2B [note 3] (born: 05/02/1985)",
}

gpt4o_enc = tiktoken.encoding_for_model("gpt-4o")
llama_enc = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
claude_enc = tiktoken.get_encoding("cl100k_base")