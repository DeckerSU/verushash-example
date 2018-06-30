import hashlib

'''
Collect your coinbase transaction (modified or not) at the front of the "transactions" list provided by the server. 
Apply a double-SHA256 hash to each transaction in the list.

Now, as long as the list has more than 1 hash remaining, go through each pair and hash them together. 
That is, concatenate the first two, double-SHA256 that, repeat for the next two, and so on. 
If you encounter an odd pair (that is, the hash list ends with a single item and no pairing), 
concatenate it with itself and hash that. Continue to do that until there is only one hash left: that is your merkle root.
'''

# Hash pairs of items recursively until a single value is obtained
def merkle(hashList):
    if len(hashList) == 1:
        return hashList[0]
    newHashList = []
    # Process pairs. For odd length, the last is skipped
    for i in range(0, len(hashList)-1, 2):
        newHashList.append(hash2(hashList[i], hashList[i+1]))
    if len(hashList) % 2 == 1: # odd, hash last item twice
        newHashList.append(hash2(hashList[-1], hashList[-1]))
    return merkle(newHashList)

def hash2(a, b):
    # Reverse inputs before and after hashing
    # due to big-endian / little-endian nonsense
    a1 = a.decode('hex')[::-1]
    b1 = b.decode('hex')[::-1]
    h = hashlib.sha256(hashlib.sha256(a1 + b1).digest()).digest()
    return h[::-1].encode('hex')

def getTxnHash(txn):
        return hashlib.sha256(hashlib.sha256(txn.decode('hex')).digest()).digest()[::-1].encode('hex')[::1]

# Block 56177
txn_pool = []
#txn_pool.insert(0, coinbase_txn)
txn_pool.append("01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff060371db000101ffffffff0200c068780400000017a91424576327d5052f8e2911a84cb634ffbb33651b05870000000000000000226a2001038b2905b17576a914f79c650ecce9537e05f28e5f2a9ccf63f65f857888acc81b375b")
txn_pool.append("0100000001d42c79c0f82d1ce1bfae9f2d69ad9c81275890c4abd37d45456b48cb0bacc61a000000006b483045022100a9791d7c2dd85caa1ae7e933ffa56ac56c01928d3bcb49e4bbddd876b5f78dac02205685df83bc82fc307ca1cc8e7680a676f1ce1972eee8bf11afbb426adbdd46b0012102af8687c88c999c162961a9d28051878b9f91f9f9b505b0b4bee4e07a676dc03affffffff0100743ba40b0000001976a914f79c650ecce9537e05f28e5f2a9ccf63f65f857888ac00000000")

txn_hashes = map(getTxnHash, txn_pool)
print txn_hashes

print "Merkle root: ", merkle(txn_hashes)
print "           :  9a8c1316d8b3dc825efcf758efde8be9af0a4de035c4506e258f9d030b71f331"
