import numpy as np

class Quaternion():

    def __init__(self,a=0,b=0,c=0,d=0):

        if (type(a) == complex) or (type(b) == complex):
            a,b,c,d = a.real, a.imag , b.real, b.imag

        if type(a) == Quaternion:
            a,b,c,d = a._q

        self._q = (a,b,c,d)

    # self[item]
    def __getitem__(self,item):
        return self._q[(item)]

    def quaternion_test(self):
        if type(self._q) != tuple:
            raise TypeError('Input is not tuple.')
        if len(self._q) != 4:
            raise ValueError('Input is not length 4.')
        for digit in self._q:
            if type(digit) != int and type(digit) != float:
                raise TypeError('Element not a real number.')
        return None

    @property
    def real(self):
        self.quaternion_test()
        return(self._q[0])

    @property
    def imag(self):
        self.quaternion_test()
        return(self._q[1])

    @property
    def jmag(self):
        self.quaternion_test()
        return(self._q[2])

    @property
    def kmag(self):
        self.quaternion_test()
        return(self._q[3])

    @property
    def scalar(self):
        self.quaternion_test()
        return(self._q[0])

    @property
    def vector(self):
        self.quaternion_test()
        return (self._q[1], self._q[2], self._q[3])

    @property
    def complex_pair(self):
        self.quaternion_test()
        return (self.real+self.imag*1j, self.jmag+self.kmag*1j)

    @property
    def matrix(self):
        self.quaternion_test()
        return np.array([[self.real+self.imag*1j, self.jmag+self.kmag*1j],
                        [-1*self.jmag+self.kmag*1j, self.real-self.imag*1j]])

    def __str__(self):
        self.quaternion_test()
        string = '('
        if self.real > 0:
            string += str(self.real)
        elif self.real < 0:
            string += str(self.real)
        else: # if element == 0
            string += str(self.real)
        if self.imag > 0:
            string += '+' + str(self.imag) + 'i'
        elif self.imag < 0:
            string += str(self.imag) + 'i'
        else: # if element == 0
            string += '+' + str(self.imag) + 'i'
        if self.jmag > 0:
            string += '+' + str(self.jmag) + 'j'
        elif self.jmag < 0:
            string += str(self.jmag) + 'j'
        else: # if element == 0
            string += '+' + str(self.jmag) + 'j'
        if self.kmag > 0:
            string += '+' + str(self.kmag) + 'k'
        elif self.kmag < 0:
            string += str(self.kmag) + 'k'
        else: # if element == 0
            string += '+' + str(self.kmag) + 'k'
        string += ')'
        return string

    def __repr__(self):
        return str(self)

    def __pos__(self):
        self.quaternion_test()
        return self

    def __neg__(self):
        self.quaternion_test()
        return Quaternion(-1*self.real, -1*self.imag, -1*self.jmag, -1*self.kmag)

    def __add__(self, other):
        self, other = test(self, other)
        #if type(other) == np.ndarray:
        #    return NotImplemented
        #else:
        return Quaternion(self.real + other.real, self.imag + other.imag, self.jmag + other.jmag, self.kmag + other.kmag)

    def __radd__(self, other):
        self, other = test(self, other)
        return other + self

    def __sub__(self, other):
        self, other = test(self, other)
        return self + -other

    def __rsub__(self, other):
        self, other = test(self, other)
        return -self + other

    def __mul__(self, other):
        self, other = test(self, other)

        a = self.real*other.real - self.imag*other.imag - self.jmag*other.jmag - self.kmag*other.kmag
        b = self.real*other.imag + self.imag*other.real + self.jmag*other.kmag - self.kmag*other.jmag
        c = self.real*other.jmag - self.imag*other.kmag + self.jmag*other.real + self.kmag*other.imag
        d = self.real*other.kmag + self.imag*other.jmag - self.jmag*other.imag + self.kmag*other.real

        # Trim floats to 14dp - handles big numbers in scientific notation
        return Quaternion(float('{:0.14e}'.format(a)), float('{:0.14e}'.format(b)), float('{:0.14e}'.format(c)), float('{:0.14e}'.format(d)))

    def __rmul__(self, other):
        self, other = test(self, other)
        return other * self

    def conjugate(self):
        self.quaternion_test()
        return Quaternion(self.real, -1*self.imag, -1*self.jmag, -1*self.kmag)

    def inverse(self):
        self.quaternion_test()
        # q_inv = q_conjugate / (q_norm)^2
        # Norm square inverse
        conj = self.conjugate()
        nsi = 1/(conj * self)[0]
        return Quaternion(nsi*conj.real, nsi*conj.imag, nsi*conj.jmag, nsi*conj.kmag)

    def __truediv__(self, other):
        self, other = test(self, other)
        return self * other.inverse()

    def __rtruediv__(self, other):
        self, other = test(self, other)
        return self.inverse() * other

    def __eq__(self, other):
        self, other = test(self, other)

        diff = self - other
        result = True
        for i in range(4):
            if diff[i] != 0:
                result = False
        return result

    def __pow__(self, n):
        self.quaternion_test()
        if type(n) != int:
            raise TypeError('Input n is not an int.')

        if n == 0:
            return Quaternion(1,0,0,0)
        elif n == 1:
            return self
        elif n == -1:
            return self.inverse()
        elif n > 1:
            qi = self
            for i in range(n-1):
                qi *= self
            return qi
        elif n < -1:
            qi = self.inverse()
            for i in range(-n-1):
                qi *= self.inverse()
            return qi

    def __abs__(self):
        self.quaternion_test()
        return np.sqrt((self.conjugate() * self)[0])

# Helper Functions

def convert(elem):
    if type(elem) == float or type(elem) == int:
    # Trim floats to 14dp - big numbers in handles scientific notation
        return Quaternion(float('{:0.14e}'.format(elem)), 0, 0, 0)
    elif type(elem) == complex:
        return Quaternion(float('{:0.14e}'.format(elem.real)), float('{:0.14e}'.format(elem.imag)), 0, 0)
    #elif type(elem) == np.ndarray:
    #    return elem
    else: # type(elem) == Quaternion:
        return Quaternion(float('{:0.14e}'.format(elem.real)), float('{:0.14e}'.format(elem.imag)), float('{:0.14e}'.format(elem.jmag)), float('{:0.14e}'.format(elem.kmag)))

def test(q, r):
    q = convert(q)
    r = convert(r)
    #if type(q) != np.ndarray:
    q.quaternion_test()
    #if type(r) != np.ndarray:
    r.quaternion_test()
    return q, r
