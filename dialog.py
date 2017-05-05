#########################################################
# CIA CREST Explorer Dialog System			#
# By Raquel Alvarez					#
#########################################################

# Output to the user the different options available
print("\n")
print("\n")
print("##################################################")
print("#	     CREST Archive Explorer		#")
print("##################################################")

print("Welcome! I am here to assist you on exploring")
print("the CREST Archive released by the CIA in Feb.")
print("2017.\n")

print("There are a total of 1,011 documents that I")
print("can help you explore today. This is just a")
print("subset of documents from the 11 million pages")
print("available in the archive.\n")

print("To better assist you with the exploration, I")
print("used a machine learning algorithm to generate")
print("clusters that group documents by what is in them.\n")

print("Below is a set of options that you may choose")
print("from:\n")

print("1. OCR new documents")
print("		To OCR new documents, just run the script")
print("		get_text.py with the PDF as an argument:")
print("		    python get_text.py input_file.pdf")
print("		This will generate a text file with the")
print("		same name as the input file.\n")

print("2. Clustering - show the top 10 terms per cluster")
print("		I have created a default of 40 clusters. This")
print("		option will display the top 10 occuring terms")
print("		for each cluster.\n")

print("3. Clustering - show the top 100 terms per cluster\n")

print("4. Clustering - show where I can find the documents")
print("			that belong to a particular cluster\n")

print("5. Clustering - show the most descriptive documents")
print("			per cluster\n")

print("5. Bonus Feature - Run a generative machine learning model")
print("			  that will generate new text from the input")
print("			  file. This is an experimental tool, and does")
print("			  require fine tuning and special hardware to")
print("			  produce meaninful output. But you are welcome")
print("			  to play around with it!\n")

print("			  This is how it works. Go into the folder")
print("			  crest_explorer/generative_model. There you")
print("			  will find two possible scripts: lstm_rnn.py")
print("			  and larger_lstm_rnn.py. This code is a slightly")
print("			  modified version of the tutorial:")
print("			  http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/")
print("			  You can run this using the following command:")
print("			      python lstm_rnn.py input_file.txt\n")

print("For more info, refer to the README.md in this folder!\n")

print("#####################################################\n")


# Take user input
option = input("Please enter the number of the option you would like to choose: ")

# Check the user input and output more information as needed
if option == '1':
	print("You chose: OCR DOCUMENTS\n")
	print("Please refer to the following folder to find the tool:")
	print("		---> tools/text_extraction <---")
	print("You can find the 
