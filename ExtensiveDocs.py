import gensim.models as gsm
from os import listdir
from os.path import isfile, join
from gensim.models.doc2vec import TaggedDocument
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import os
import docx2txt
from docx2txt import process
import gensim


#convert word corpus to txt corpus
folderDirectory = filedialog.askdirectory()
#print(folderDirectory)
for filename in os.listdir(folderDirectory):
    f = os.path.join(folderDirectory, filename)
        # checking if it is a file
    if os.path.isfile(f):
        MY_TEXT = docx2txt.process(f)
        #print(f)
        #print(filename)

        newFileName = filename.rsplit('.', 1)[0]

        directory = "Corpus"
        parent_dir = folderDirectory
        path = os.path.join(parent_dir, directory)
        isExist = os.path.exists(path)
        #print(isExist)
        if(isExist == False):
            (os.makedirs(path))
        else:
            print("File already exists")

        newTxt = folderDirectory + "/Corpus/" + newFileName + ".txt"

        with open(newTxt, "w") as text_file:
            print(MY_TEXT, file=text_file)

#testing

#path to the input corpus files
train_corpus = folderDirectory + "/Corpus"

#tagging the text files
class DocIterator(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[self.labels_list[idx]])

docLabels = [f for f in listdir(train_corpus) if f.endswith('.txt')]
#print(docLabels)
length = len(docLabels)
data = []
for doc in docLabels:
    data.append(open(join(train_corpus, doc), 'r', encoding = 'iso-8859-1', errors = 'ignore').read())
    
it = DocIterator(data, docLabels)

#train doc2vec model
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
#model = gsm.Doc2Vec(vector_size=50, window=5, min_count=1, workers=5,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(it)
model.train(it, total_examples=len(doc), epochs=30)
print(model.epochs)

#Create a Model Folder

def createModelFolder():
    directorymod = "Model"
    parent_dir_mod = folderDirectory
    pathMod = os.path.join(parent_dir_mod, directorymod)
    isExist = os.path.exists(pathMod)

    #print(pathMod)

    #print(isExist)
    if(isExist == False):
        (os.makedirs(pathMod))
    else:
        print("File already exists")

    return pathMod

createModelFolder()

model.save(createModelFolder() + "/paper.model")

#print(model.wv.most_similar("bad"))

print("model is saved")

#part2
#loading the model
model= createModelFolder() + "/paper.model"
m=gsm.Doc2Vec.load(model)
print("model is loaded")

#part 3
#path to test files
#test_paper=askopenfilename()

path = askopenfilename()
text = process(path)
with open(os.path.basename(path) + '.txt', 'w') as f:
    print(text)

newTxt = path.rsplit('/', 1)[1]
filename = newTxt.rsplit('.', 1)[0]
#print(newTxt)
newTxt = folderDirectory + "/Corpus/" + filename + ".txt"
#print(newTxt)

with open(newTxt, "w") as text_file:
    print(text, file=text_file)


#print(test_paper)
new_test = open(join(path), 'r', encoding = 'iso-8859-1', errors = 'ignore').read().split()
#print(new_test)
inferred_docvec = m.infer_vector(new_test)
#m.docvecs.most_similar([inferred_docvec], topn=3)
print('%s:\n %s' % (model, m.dv.most_similar(positive=[inferred_docvec], topn=length)))