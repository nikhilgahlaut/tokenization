import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
stop_words = set(stopwords.words('english')) 

# Dummy text 
txt = "It is at school that I learn so much. I learn so many different subjects." \
"There are many interesting co-curricular activities too." \
" I also participate in extra-curricular activities like music, dance, art, craft and plays."\
 "I also get the opportunity to take part in sports and games on the sports grounds at school."\
 "I am part of the school’s volleyball team."\
 "Our teachers teach us with a great deal of care and patience, and I am very grateful to the teachers."\
 " At school I have many good friends."\
 "We study, play and eat together." \
"I love my school very much."\

# sent_tokenize is one of instances of 
# PunktSentenceTokenizer from the nltk.tokenize.punkt module 

tokenized = sent_tokenize(txt) 
for i in tokenized: 
	
	# Word tokenizers is used to find the words 
	# and punctuation in a string 
	wordsList = nltk.word_tokenize(i) 

	# removing stop words from wordList 
	wordsList = [w for w in wordsList if not w in stop_words] 

	# Using a Tagger. Which is part-of-speech 
	# tagger or POS-tagger. 
	tagged = nltk.pos_tag(wordsList) 

	print(tagged) 
