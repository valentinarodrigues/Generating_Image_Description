from pickle import load
from keras.preprocessing.text import Tokenizer
from pickle import dump
# load doc into memory
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return sorted(set(dataset))

"""
def train_test_split(dataset):
	ordered = sorted(set(dataset))
	return sorted(set(ordered[500:1500])), sorted(set(ordered[1500:2500]))
"""
def train_test_split(dataset):
	ordered = sorted(set(dataset))
	return sorted(set(ordered[0:5000])), sorted(set(ordered[5000:6000]))

def load_clean_descriptions(filename, dataset):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		image_id, image_desc = tokens[0], tokens[1:]
		if image_id in dataset:
			descriptions[image_id] = 'startseq ' + ' '.join(image_desc) + ' endseq'
	return descriptions

def create_tokenizer(descriptions):
	lines = list(descriptions.values())
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer


filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
dataset = load_set(filename)
train, test = train_test_split(dataset)
#print(train)
descriptions = load_clean_descriptions('descriptions.txt', train)
#print(descriptions)
tokenizer = create_tokenizer(descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))


"""
def load_photo_features(filename, dataset):
	all_features = load(open(filename, 'rb'))
	features = {k: all_features[k] for k in dataset}
	return features
print('Dataset: %d' % len(dataset))
# train-test split
train, test = train_test_split(dataset)
print('Train=%d, Test=%d' % (len(train), len(test)))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))
# photo features
train_features = load_photo_features('features_extracted.pkl', train)
test_features = load_photo_features('features_extracted.pkl', test)
print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizertrain.pkl', 'wb'))
tokenizer = create_tokenizer(test_descriptions)
dump(tokenizer, open('tokenizertest.pkl', 'wb'))
"""
