
text="data/200x150Tree.asc"

# reads in a text file that determines the environmental grid setup from ABM
f = open(text, 'r')
body = f.readlines()
width = body[0][-4:-1]  # last 4 characters of line that contains the 'width' value
height = body[1][-5:-1]
abody = body[7:]  # ASCII file with a header
f.close()
abody = reversed(abody)
cells = []
with open("200x150Tree.asc","w") as writer:
    for line in abody:
        theline = line.split(" ")
        thelinenew = []
        for i in range(0,len(theline)):
            if theline[i] == '-3.4028234663852885981e+38':
                # theline[i] = '2'
                # theline[i+1] = " "
                thelinenew.append('2')
                thelinenew.append(" ")
            else:
                # theline[i] = '0'
                # theline[i+1] = " "
                thelinenew.append('0')
                thelinenew.append(" ")
        thelinenew.append("\n")
        writer.writelines(thelinenew)

    writer.close()
