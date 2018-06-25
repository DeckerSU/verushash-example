<?php

/*

Collect your coinbase transaction (modified or not) at the front of the "transactions" list provided by the server. 
Apply a double-SHA256 hash to each transaction in the list.

Now, as long as the list has more than 1 hash remaining, go through each pair and hash them together. 
That is, concatenate the first two, double-SHA256 that, repeat for the next two, and so on. 
If you encounter an odd pair (that is, the hash list ends with a single item and no pairing), 
concatenate it with itself and hash that. Continue to do that until there is only one hash left: that is your merkle root.

*/

$coinbase_data = pack("H*", "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0603f2a3000142ffffffff020080d1f00800000017a914007c2479d28777e55410475b6bb4aec6806e3d858700000000000000002c6a2a0103369803b1752103cdc323d03141e97a933dbaae3853611ae2ea090b9209fa12347598d0d0347579ac9cc8295b");
$hash = hash("sha256", $coinbase_data);
$hash = hash("sha256", pack("H*", $hash));
var_dump($hash);
$hash = bin2hex(strrev(pack("H*", $hash)));

var_dump($hash);
var_dump("6e2964309c1e66ac7cb0bd67ff18cd1c5362772dc1c37d8fa3d06347ccfd4361");

?>