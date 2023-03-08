from reedsolo import RSCodec, ReedSolomonError

class ReedSolomon:
    @staticmethod
    def encode(message, errorBits=10, intArray=True):
        rsc = RSCodec(errorBits)
        encoded = rsc.encode(bytearray(message))
        if (intArray):
            return [x for x in encoded]
        return encoded
    
    @staticmethod
    def decode(message, errorBits=10):
        try:
            rsc = RSCodec(errorBits)
            decoded = rsc.decode(bytearray(message))[0]
            return [x for x in decoded]
        except ReedSolomonError:
            print("Reed Solomon Error")
            return None

        