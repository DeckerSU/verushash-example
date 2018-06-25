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
        return hashlib.sha256(hashlib.sha256(txn.decode('hex')).digest()).digest().encode('hex')[::1]

txn_pool = []
'''
txn_pool.append("")
txn_pool.append("0100000001440d795fa6267cbae00ae18e921a7b287eaa37d7f41b96ccbc61ef9a323a003d010000006a47304402204137ef9ca79bcd8a953c0def89578838bbe882fe7814d6a7144eaa25ed156f66022043a4ab91a7ee3bf58155d08e5f3f221a783f645daf9ac54fed519e18ca434aea012102965a03e05b2e2983c031b870c9f4afef1141bf30dc5bb993197ee4a52f1443e0feffffff0200a3e111000000001976a914f1cfa585d096ea3c759940d7bacd8c7259bbd4d488ac4e513208000000001976a9146701f2540186d4135eec14dad6cb25bf757fc43088accbd50600")
txn_pool.append("0100000001517063b3d932693635999b8daaed9ebf020c66c43abf504f3043850bca5a936d010000006a47304402207473cda71b68a414a53e01dc340615958d0d79dd67196c4193a0ebcf0d9f70530220387934e7317b60297f5c6e0ca4bf527faaad830aff45f1f5522e842595939e460121031d53a2c228aedcde79b6ccd2e8f5bcfb56e2046b4681c4ea2173e3c3d7ffc686ffffffff0220bcbe00000000001976a9148cc3704cbb6af566598fea13a3352b46f859581188acba2cfb09000000001976a914b59b9df3700adae0ea819738c89db3c2af4e47d188ac00000000")
'''

coinbase_txn = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0603f2a3000142ffffffff020080d1f00800000017a914007c2479d28777e55410475b6bb4aec6806e3d858700000000000000002c6a2a0103369803b1752103cdc323d03141e97a933dbaae3853611ae2ea090b9209fa12347598d0d0347579ac9cc8295b"
# hash = 6143fdcc4763d0a38f7dc3c12d7762531ccd18ff67bdb07cac661e9c3064296e = 0x6e2964309c1e66ac7cb0bd67ff18cd1c5362772dc1c37d8fa3d06347ccfd4361
# print hashlib.sha256(coinbase_txn.decode('hex')).digest().encode('hex')

print "Tx hash: ", getTxnHash(coinbase_txn)
print "Tx hash: ", getTxnHash(coinbase_txn).decode('hex')[::-1].encode('hex')

txn_pool.insert(0, coinbase_txn)
txn_hashes = map(getTxnHash, txn_pool)

print "Merkle root: ", merkle(txn_hashes).decode('hex')[::-1].encode('hex')
