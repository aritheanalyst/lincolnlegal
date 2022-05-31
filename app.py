#**************** IMPORT PACKAGES ********************
import flask
from flask import render_template, jsonify, Flask, redirect, url_for, request, flash
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import numpy as np
import pytesseract as pt
import pdf2image
from fpdf import FPDF
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import os
import pdfkit
import yake
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModel, AutoConfig
from summarizer import Summarizer,TransformerSummarizer
from transformers import pipelines
#nltk.download('punkt')

print("lets go")


app = flask.Flask(__name__)
app.config["DEBUG"] = True
UPLOAD_FOLDER = './pdfs'

ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#***************** FLASK *****************************
CORS(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



#model_name = 'laxya007/gpt2_legal'
#model_name = 'facebook/bart-large-cnn'
model_name = 'nlpaueb/legal-bert-base-uncased'


#The setup of huggingface.co

print("lets go")

custom_config = AutoConfig.from_pretrained(model_name)
print("lets go")
custom_config.output_hidden_states=True
print("lets go")
custom_tokenizer = AutoTokenizer.from_pretrained(model_name)
print("lets go")
#custom_model = AutoModel.from_pretrained(model_name, config=custom_config)
print("lets go")
bert_legal_model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
print('Using model {}\n'.format(model_name))



# main index page route
@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@cross_origin()
@app.route('/results')
def results():
    return render_template('results.html')



@app.route('/predict', methods=['GET', 'POST'])
def uploads():
    if request.method == 'GET':
        # Get the file from post request

        numsent = int(request.args['number'])
        text = str(request.args['text'])
        content = text


        summary_text = ""
        for i, paragraph in enumerate(content.split("\n\n")):
            
            paragraph = paragraph.replace('\n',' ')
            paragraph = paragraph.replace('\t','')
            paragraph = ' '.join(paragraph.split())
            # count words in the paragraph and exclude if less than 4 words
            tokens = word_tokenize(paragraph)
            # only do real words
            tokens = [word for word in tokens if word.isalpha()]
            # print("\nTokens: {}\n".format(len(tokens)))
            # only do sentences with more than 1 words excl. alpha crap
            if len(tokens) <= 1:
                continue
            # Perhaps also ignore paragraphs with no sentence?
            sentences = sent_tokenize(paragraph)
            
            paragraph = ' '.join(tokens)

            print("\nParagraph:")
            print(paragraph+"\n")
            # T5 needs to have 'summarize' in order to work:
            # text = "summarize:" + paragraph
            text = paragraph
            
            summary = bert_legal_model(text,  min_length = 8, ratio = 0.05)
            # summary = tokenizer_t5.decode(summary_ids[0], skip_special_tokens=True)
            summary_text += str(summary) + "\n\n"
            print("Summary:")
            print(summary)

        content2 = content.replace('\n',' ')
        content2 = content2.replace('\t','')
        summary = bert_legal_model(content2, min_length = 8, num_sentences=25)
        


        # write all to file for inspection and storage
        all_text = "The Summary-- " + str(summary) + "\n\n\n" \
            + "The Larger Summary-- " + str(summary_text)
            

        all_text2 = all_text.encode('latin-1', 'replace').decode('latin-1')
        all_text2 = all_text2.replace('?','.')
        all_text2 = all_text2.replace('\n',' ')
        all_text2 = all_text2.replace('..','.')
        all_text2 = all_text2.replace(',.',',')
        all_text2 = all_text2.replace('-- ','\n\n\n')

        pdf = FPDF()  

        # Add a page
        pdf.add_page()

        pdf.set_font("Times", size = 12)

        # open the text file in read mode
        f = all_text2

        # insert the texts in pdf
        pdf.multi_cell(190, 10, txt = f, align = 'C')


        # save the pdf with name .pdf
        pdf.output("./static/legal.pdf")  
        all_text

        
        return render_template('results.html')
    return None




@app.route('/predictpdf', methods=['GET', 'POST'])
def uploads2():
    if request.method == 'POST':
        # Get the file from post request

        numsent = int(request.args['number'])
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = "legal.pdf"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        f = request.files['file']
        f.save(secure_filename(f.filename))


        path = os.getcwd()
        folder_name = 'pdfs'
        path = os.path.join(path, folder_name) 

        list_of_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if(file.endswith(".pdf")):
                    # print(os.path.join(root,file))
                    list_of_files.append(os.path.join(root,file))

        print("\nProcessing {} files...\n".format(len(list_of_files)))
        total_pages = 0

        for filename in list_of_files:
            print(filename)
            file = os.path.splitext(os.path.basename(filename))[0]
            pages = pdf2image.convert_from_path(pdf_path=filename, dpi=400, size=(1654,2340))
            total_pages += len(pages)
            print("\nProcessing the next {} pages...\n".format(len(pages)))

            # Then save all pages as images and convert them to text except the last page
            # TODO: create this as a function
            content = ""
            dir_name = 'images/' + file + '/' 
            os.makedirs(dir_name, exist_ok=True)
            # If folder doesn't exist, then create it.
            for i in range(len(pages)-1):
                pages[i].save(dir_name + str(i) + '.jpg')
                # OCR the image using Google's tesseract
                content += pt.image_to_string(pages[i])

            summary_text = ""
            for i, paragraph in enumerate(content.split("\n\n")):
                
                paragraph = paragraph.replace('\n',' ')
                paragraph = paragraph.replace('\t','')
                paragraph = ' '.join(paragraph.split())
                # count words in the paragraph and exclude if less than 4 words
                tokens = word_tokenize(paragraph)
                # only do real words
                tokens = [word for word in tokens if word.isalpha()]
                # print("\nTokens: {}\n".format(len(tokens)))
                # only do sentences with more than 1 words excl. alpha crap
                if len(tokens) <= 1:
                    continue
                # Perhaps also ignore paragraphs with no sentence?
                sentences = sent_tokenize(paragraph)
                
                paragraph = ' '.join(tokens)

                print("\nParagraph:")
                print(paragraph+"\n")
                # T5 needs to have 'summarize' in order to work:
                # text = "summarize:" + paragraph
                text = paragraph
                
                summary = bert_legal_model(text,  min_length = 8, ratio = 0.05)
                # summary = tokenizer_t5.decode(summary_ids[0], skip_special_tokens=True)
                summary_text += str(summary) + "\n\n"
                print("Summary:")
                print(summary)

            content2 = content.replace('\n',' ')
            content2 = content2.replace('\t','')
            summary = bert_legal_model(content2, min_length = 8, num_sentences=25)
            


            # write all to file for inspection and storage
            all_text = "The Summary-- " + str(summary) + "\n\n\n" \
                + "The Larger Summary-- " + str(summary_text)
                

            all_text2 = all_text.encode('latin-1', 'replace').decode('latin-1')
            all_text2 = all_text2.replace('?','.')
            all_text2 = all_text2.replace('\n',' ')
            all_text2 = all_text2.replace('..','.')
            all_text2 = all_text2.replace(',.',',')
            all_text2 = all_text2.replace('-- ','\n\n\n')

            pdf = FPDF()  

            # Add a page
            pdf.add_page()

            pdf.set_font("Times", size = 12)

            # open the text file in read mode
            f = all_text2

            # insert the texts in pdf
            pdf.multi_cell(190, 10, txt = f, align = 'C')


            # save the pdf with name .pdf
            pdf.output("./static/legal.pdf")  
            all_text

            
        return render_template('results.html')
    return None


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)


