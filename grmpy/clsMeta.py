''' Meta class for the moment models.
'''

# standard library
import pickle as pkl
try:
   import cPickle as pkl
except:
   import pickle as pkl

class metaCls(object):
    
    def __init__(self):
        
        pass
    
    ''' Meta methods.
    '''
    def get_status(self):
        ''' Get status of class instance.
        '''
        
        return self.isLocked

    def lock(self):
        ''' Lock class instance.
        '''
        # Antibugging.
        assert (self.get_status() == False)

        # Update class attributes.
        self.isLocked = True
        
        # Finalize.
        self._derivedAttributes()
        
        self._checkIntegrity()
    
    def unlock(self):
        ''' Unlock class instance.
        '''
        # Antibugging.
        assert (self.get_status() == True)

        # Update class attributes.
        self.isLocked = False

    def getAttr(self, key):
        ''' Get attributes.
        '''
        # Antibugging.
        assert (self.get_status() == True)
        assert (self._checkKey(key) == True)
        
        # Finishing.
        return self.attr[key]

    def setAttr(self, key, value):
        ''' Get attributes.
        '''
        # Antibugging.
        assert (self.get_status() == False)
        assert (self._checkKey(key) == True)
        
        # Finishing.
        self.attr[key] = value
    
    def store(self, fileName):
        ''' Store class instance.
        '''
        # Antibugging.
        assert (self.get_status() == True)
        assert (isinstance(fileName, str))
        
        # Store.
        pkl.dump(self, open(fileName, 'wb'))
        
    def _checkKey(self, key):
        ''' Check that key is present.
        '''        
        # Check presence.
        assert (key in self.attr.keys())
        
        # Finishing.
        return True
    
    def _derivedAttributes(self):
        ''' Calculate derived attributes.
        '''
        
        pass
    
    def _checkIntegrity(self):
        ''' Check integrity of class instance.
        '''
        
        pass