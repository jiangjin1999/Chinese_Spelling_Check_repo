import logging
import codecs
from bs4 import BeautifulSoup



def read_langs(file_name):

    logging.info(("Reading lines from {}".format(file_name)))
    total_data=[]

    with codecs.open(file_name, "r", "utf-8") as file:

        data = file.read()
        # data = data[0:2116]
        soup = BeautifulSoup(data, 'html.parser')
        results = soup.find_all('sentence')
        for item in results:

            text = item.find("text").text.strip()
            mistakes = item.find_all("mistake")

            error_informations = []
            for mistake in mistakes:
                location = mistake.find("location").text.strip()
                wrong =  mistake.find("wrong").text.strip()
                right = mistake.find("correction").text.strip()
                error_infor_temp = []
                error_infor_temp.append(location)
                error_infor_temp.append(',')
                error_infor_temp.append(wrong)
                error_infor_temp.append(',')
                error_infor_temp.append(right)
                error_infor_temp.append(';')
                error_informations.append(error_infor_temp)
                if text[int(location)-1] != wrong:
                    print("The character of the given location does not equal to the real character")
            sentence = text
            if len(sentence) < 511:
                with open('data/Auto_Gener_Data/train.txt', 'a', encoding='utf-8') as f:
                    f.write(sentence)
                    f.write("\n")
                    for i in range(0, len(error_informations)):
                        f.writelines(error_informations[i])
                    f.write("\n")



# ----------Sentence-Level don't need tag---------
#           sen = list(text)
#           tags = ["0" for _ in range(len(sen))]
#           for i in locations:
#               tags[i - 1] = "1"

#            total_data.append([" ".join(sentence), " ".join(tags)])

    return total_data