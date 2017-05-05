# xml structure: <id> </id> <article> </article> <summary> </summary> 

import os,sys
import fnmatch
from lxml import etree
import codecs

def convertToSolrXML(content,counter):

        c='<add><doc>\n\t<field name="id">'

        c=c+str(counter)

        c=c+'</field>\n\t<field name="article">'

        try:

            article=""+content.split("Article:")[1].split("Summary:")[0].strip(' \t\n\r')+""

            summarysentences=content.split("Summary:")[1].strip(' \t\n\r').split("\n")

            summary=""

            for summarysentence in summarysentences:

                summary+="----"+summarysentence+"----"

        except (ValueError,IndexError):

            print "couldn't get content or summary"

            return None

        c=c+article+'</field>\n\t<field name="summary">'

        c=c+summary

        c=c+'</field>\n</doc></add>'

        #sanitizing the xml before returning

        try:

                parser = etree.XMLParser(recover=True) # recover from bad characters.

                root = etree.fromstring(c, parser=parser)

                return etree.tostring(root,encoding='UTF-8')

        except:

                print "couldn't sanitize",os.path.abspath(fileName)

                return None



def main():

        inpDir=sys.argv[1]

        print "file searching started"

        files=[x for x in os.listdir(inpDir) if x.endswith("txt")]

        f1=open("FAILURES","w")

        for index,item in enumerate(files):

                print index,os.path.join(inpDir,item)[:-4]+".xml","out of",len(files),"being processed"

                #print codecs.open(os.path.join(inpDir,item),encoding='utf-8',mode='r').read()

                c=convertToSolrXML(codecs.open(os.path.join(inpDir,item),encoding='utf-8',mode='r').read(),index+1)

                if c:

                        with open(os.path.join(inpDir,"xmls",item)[:-4]+".xml","w") as f:

                                f.write(c)

                else:

                        print "problem converting to xml file"

                        f1.write(item+"\n")

        f1.close()



if __name__ == "__main__":

        main()
