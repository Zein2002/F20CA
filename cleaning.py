import json

#set file source
f = open(r"C:\Users\Áron\Google Drive\UNI\Y4S2\Conversational Agents\F20CA\questions.json", "r")
data = json.load(f)

#file to return and temp storage for questions
res = []
temp = []

#go through all games
for game in data.get("games"):
    for q in game.get("questions"):
        #if a question doesnt end in a "?" it requires the answers to be read
        if q.get("question")[-1] == "?":
            temp.append(q)
    res += temp
    temp = []

with open('questions_clean.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)