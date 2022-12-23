# Implement the Bloom filter.
# Use: http://www.stopforumspam.com/downloads/listed_username_30.zip as set S.
# This is a set of usernames known to be spam for the last 30 days.
# Select a proper hashing memory size (n) and find the optimal number of hash functions (k).
# Use the spam usernames for the last 365 days: http://www.stopforumspam.com/downloads/listed_username_365.zip as your stream.

from bitarray import bitarray
import mmh3
import math

# Bloom Filter using MurmurHash function
class BloomFilter:
    def __init__(self, m, fp):
        # hashing memory size n
        self.hashSize = self.optimal_hash_size(m, fp)
        # number of hash functions k
        self.numFunctions = self.optimal_num_hashes(m, self.hashSize)
        # initialize array of size n to all 0s
        self.bitArray = bitarray(self.hashSize)
        self.bitArray.setall(0)

    def filter_train(self, item):
        for i in range(self.numFunctions):
            index = mmh3.hash(item, i) % self.hashSize
            self.bitArray[index] = True

    def filter_test(self, item):
        for i in range(self.numFunctions):
            index = mmh3.hash(item, i) % self.hashSize
            if self.bitArray[index] == False:
                return False # have not seen
        return True # have seen

    # Consider: |S| = m, |B| = n, |hashes| = k
    def optimal_hash_size(self, m, fp):
        # m: number of stored items
        # n: bit array size
        # P of false positive = 1 - e^(-m/n) = Fraction of 1s in the array B
        # return n
        return int( -m/math.log(1-fp) )

    def optimal_num_hashes(self, m, n):
        # m: number of stored items
        # n: bit array size
        # return k
        return int( (n/m) * math.log(2) )


# create a list with seen spam email addresses
have_seen = []
with open('listed_username_30.txt') as f30:
    lines30 = f30.readlines()
for line in lines30:
    have_seen.append(line.strip())

# create filter object
filter = BloomFilter(len(have_seen), 0.1)
print('proper hashing memory size: ', filter.hashSize)
print('optimal number of hash functions: ', filter.numFunctions)

for email in have_seen:
    filter.filter_train(email)
with open('listed_username_365.txt') as f365:
    lines365 = f365.readlines()
false_positive = 0
total = 0
for line in lines365:
    line = line.strip()
    total += 1
    if filter.filter_test(line) == True and line not in have_seen:
        false_positive += 1
print('false positive rate:', round(false_positive/total, 2))
