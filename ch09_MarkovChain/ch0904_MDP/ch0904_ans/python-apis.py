import re # python-apis.py

text1 = "#this is a 5string"
print("length:",len(text1))

text2 = text1.split(' ')
set([w.lower() for w in text2])
words2 = [w for w in text2 if w.startswith('#')]
words3 = [w for w in text2 if re.search('^[A-Za-z0-9_]+', w)]

print("words2:",words2)
print("words3:",words3)

