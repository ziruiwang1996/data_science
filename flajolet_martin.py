# Flajolet-Martin (FM) algorithm. Count the number of distinct quotes (quotes
# are denoted with lines that start with Q) in the MemeTracker dataset (all files)
# https://snap.stanford.edu/data/memetracker9.html
# Output: the estimated number from FM
# In your implementation, use the method discussed in Section 4.4.3 to provide more accurate results.

def append(lines):
    for line in lines:
        if line.startswith("Q"):
            line = line[1:] # remove Q
            line = line.strip()
            stream.append(line)

def FlajoletMartin(stream):
    maxR = 0
    for element in stream:
        value = bin(hash(element)%1024)[2:]

        bitsStr = str(value)
        R = 0
        for i in range(len(bitsStr)-1, 0, -1):
            if bitsStr[i] == '0' :
                R += 1
            else:
                break

        if R > maxR:
            maxR = R
    return "Estimated distinct elements are " + str(2**maxR)

with open("quotes_2008-08.txt") as f08:
    lines08 = f08.readlines()
with open("quotes_2008-09.txt") as f09:
    lines09 = f09.readlines()
with open("quotes_2008-10.txt") as f10:
    lines10 = f10.readlines()
#with open("quotes_2008-11.txt") as f11:
    #lines11 = f11.readlines()
#with open("quotes_2008-12.txt") as f12:
    #lines12 = f12.readlines()
#with open("quotes_2009-01.txt") as f01:
    #lines01 = f01.readlines()
#with open("quotes_2009-02.txt") as f02:
    #lines02 = f02.readlines()
#with open("quotes_2009-03.txt") as f03:
    #lines03 = f03.readlines()
#with open("quotes_2009-04.txt") as f04:
    #lines04 = f04.readlines()
stream =[]
append(lines08)
append(lines09)
append(lines10)
#append(lines11)
#append(lines12)
#append(lines01)
#append(lines02)
#append(lines03)
#append(lines04)
print(FlajoletMartin(stream))

#### Due to RAM limitation, files in comment were not included for distinct elements counting
