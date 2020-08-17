import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I like New York in Autumn.")
s = []
for t in doc:
    s.append(t.text)
print(" ".join(s))

# name = {"ab", "bc", "zha", "ki"}
# name_dic = {file: [] for file in name}
# print(name_dic)
# s1 = 4
# s2 = 9
# print("sadfjl {} {}".format(s1, s2))