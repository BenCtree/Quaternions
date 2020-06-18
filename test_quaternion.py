import unittest
import numpy as np
from quaternion import Quaternion as qt

class TestQuaternion(unittest.TestCase):

    def setUp(self):

        # Proper quaternions
        self.q0 = qt(0,0,0,0)
        self.q1 = qt(1,2,3,4)
        self.q2 = qt(1.1,2.2,3.3,4.4)
        self.q3 = qt(1e-14, 1e-15, 1e-16, 1e-17)
        self.q4 = qt(1e14, 1e15, 1e16, 1e17)

        # Improper quaternions
        self.q5 = qt((1,2,3,4))
        self.q6 = qt([1,2,3,4])
        self.q7 = qt(1,'a',3,4)
        self.q8 = qt('hello')

        # For Arithmetic
        self.q = qt(1,2,3,4)
        self.r = qt(5,6,7,8)
        self.t = qt(9,10,11,12)
        self.s = qt(-3,-7,-4,8)
        
    def test_quaternion_test(self):

        # Test int and float tuples
        self.assertEqual(qt.quaternion_test(self.q0), None)
        self.assertEqual(qt.quaternion_test(self.q1), None)
        self.assertEqual(qt.quaternion_test(self.q2), None)
        self.assertEqual(qt.quaternion_test(self.q3), None)
        self.assertEqual(qt.quaternion_test(self.q4), None)
        self.assertEqual(qt.quaternion_test(self.q0), None)

        # Test non-tuples
        with self.assertRaises(TypeError):
            qt.quaternion_test(self.q5)
        with self.assertRaises(TypeError):
            qt.quaternion_test(self.q6)
        with self.assertRaises(TypeError):
            qt.quaternion_test(self.q7)
        with self.assertRaises(TypeError):
            qt.quaternion_test(self.q8)
        
        # No need to test wrong length tuple
        # as constructor ensures tuple is length 4

    def test_attributes(self):
        
        self.assertEqual(self.q0.real, 0)
        self.assertEqual(self.q1.real, 1)
        self.assertEqual(self.q2.real, 1.1)
        self.assertEqual(self.q3.real, 1e-14)
        self.assertEqual(self.q4.real, 1e14)

        self.assertEqual(self.q0.imag, 0)
        self.assertEqual(self.q1.imag, 2)
        self.assertEqual(self.q2.imag, 2.2)
        self.assertEqual(self.q3.imag, 1e-15)
        self.assertEqual(self.q4.imag, 1e15)

        self.assertEqual(self.q0.jmag, 0)
        self.assertEqual(self.q1.jmag, 3)
        self.assertEqual(self.q2.jmag, 3.3)
        self.assertEqual(self.q3.jmag, 1e-16)
        self.assertEqual(self.q4.jmag, 1e16)

        self.assertEqual(self.q0.kmag, 0)
        self.assertEqual(self.q1.kmag, 4)
        self.assertEqual(self.q2.kmag, 4.4)
        self.assertEqual(self.q3.kmag, 1e-17)
        self.assertEqual(self.q4.kmag, 1e17)

        self.assertEqual(self.q0.scalar, 0)
        self.assertEqual(self.q1.scalar, 1)
        self.assertEqual(self.q2.scalar, 1.1)
        self.assertEqual(self.q3.scalar, 1e-14)
        self.assertEqual(self.q4.scalar, 1e14)

        self.assertEqual(self.q0.vector, (0,0,0))
        self.assertEqual(self.q1.vector, (2,3,4))
        self.assertEqual(self.q2.vector, (2.2,3.3,4.4))
        self.assertEqual(self.q3.vector, (1e-15, 1e-16, 1e-17))
        self.assertEqual(self.q4.vector, (1e15, 1e16, 1e17))

        self.assertEqual(self.q0.complex_pair, (0+0j, 0+0j))
        self.assertEqual(self.q1.complex_pair, (1+2j,3+4j))
        self.assertEqual(self.q2.complex_pair, (1.1+2.2j,3.3+4.4j))
        self.assertEqual(self.q3.complex_pair, (1e-14+1e-15j, 1e-16+1e-17j))
        self.assertEqual(self.q4.complex_pair, (1e14+1e15j, 1e16+1e17j))

        np.testing.assert_array_equal(self.q0.matrix, np.array([[ 0.+0.j,  0.+0.j], [-0.+0.j,  0.-0.j]]))
        np.testing.assert_array_equal(self.q1.matrix, np.array([[ 1.+2.j,  3.+4.j], [-3.+4.j,  1.-2.j]]))
        np.testing.assert_array_equal(self.q2.matrix, np.array([[ 1.1+2.2j,  3.3+4.4j], [-3.3+4.4j,  1.1-2.2j]]))
        np.testing.assert_array_equal(self.q3.matrix, np.array([[ 1e-14+1e-15j,  1e-16+1e-17j], [-1e-16+1e-17j,  1e-14-1e-15j]]))
        np.testing.assert_array_equal(self.q4.matrix, np.array([[ 1e14+1e15j,  1e16+1e17j], [-1e16+1e17j,  1e14-1e15j]]))

    def test_pos(self):
        self.assertEqual(qt.__pos__(self.q0), self.q0)
        self.assertEqual(qt.__pos__(self.q1), self.q1)
        self.assertEqual(qt.__pos__(self.q2), self.q2)
        self.assertEqual(qt.__pos__(self.q3), self.q3)
        self.assertEqual(qt.__pos__(self.q4), self.q4)

        self.assertEqual(+self.q0, self.q0)
        self.assertEqual(+self.q1, self.q1)
        self.assertEqual(+self.q2, self.q2)
        self.assertEqual(+self.q3, self.q3)
        self.assertEqual(+self.q4, self.q4)

    def test_neg(self):
        self.assertEqual(qt.__neg__(self.q0), -self.q0)
        self.assertEqual(qt.__neg__(self.q1), -self.q1)
        self.assertEqual(qt.__neg__(self.q2), -self.q2)
        self.assertEqual(qt.__neg__(self.q3), -self.q3)
        self.assertEqual(qt.__neg__(self.q4), -self.q4)
        self.assertEqual(qt.__neg__(qt(-1,2,-3,4)), qt(1,-2,3,-4))

    def test_add_radd(self):
        # Add quaternions
        self.assertEqual(self.q1 + self.q2, qt(2.1,4.2,6.3,8.4))
        self.assertEqual(self.q2 + self.q1, qt(2.1,4.2,6.3,8.4))
        # Add int
        self.assertEqual(self.q1 + 1, qt(2,2,3,4))
        self.assertEqual(1 + self.q1, qt(2,2,3,4))
        # Add float
        self.assertEqual(self.q1 + 1.1, qt(2.1,2,3,4))
        self.assertEqual(1.1 + self.q1, qt(2.1,2,3,4))
        # Add complex
        self.assertEqual(self.q1 + 1+2j, qt(2,4,3,4))
        self.assertEqual(1+2j + self.q1, qt(2,4,3,4))
        # Small
        self.assertEqual(self.q3 + qt(0,0,0,1), qt(1e-14, 1e-15, 1e-16, 1+1e-17))
        self.assertEqual(qt(0,0,0,1) + self.q3, qt(1e-14, 1e-15, 1e-16, 1+1e-17))
        # Big
        self.assertEqual(self.q4 + qt(0,0,0,1), qt(1e14, 1e15, 1e16, 1+1e17))
        self.assertEqual(qt(0,0,0,1) + self.q4, qt(1e14, 1e15, 1e16, 1+1e17))

    def test_sub_rsub(self):
        # Subtract quaternions
        np.testing.assert_almost_equal(self.q1 - self.q2, qt(-0.1,-0.2,-0.3,-0.4), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2 - self.q1, qt(0.1,0.2,0.3,0.4), decimal=14, err_msg='', verbose=True)
        # Sub int
        self.assertEqual(self.q1 - 1, qt(0,2,3,4))
        self.assertEqual(1 - self.q1, qt(0,-2,-3,-4))
        # Sub float
        np.testing.assert_almost_equal(self.q1 - 1.1, qt(-0.1,2,3,4), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(1.1 - self.q1, qt(0.1,-2,-3,-4), decimal=14, err_msg='', verbose=True)
        # Sub complex
        self.assertEqual(self.q1 - (1+2j), qt(0,0,3,4))
        self.assertEqual((1+2j) - self.q1, qt(0,0,-3,-4))
        # Small
        self.assertEqual(self.q3 - qt(0,0,0,1), qt(1e-14, 1e-15, 1e-16, 1e-17-1))
        self.assertEqual(qt(0,0,0,1) - self.q3, qt(-1e-14, -1e-15, -1e-16, 1-1e-17))
        # Big
        self.assertEqual(self.q4 - qt(0,0,0,1), qt(1e14, 1e15, 1e16, 1e17-1))
        self.assertEqual(qt(0,0,0,1) - self.q4, qt(-1e14, -1e15, -1e16, 1-1e17))
    
    def test_mul_rmul(self):
        # Mul quaternions
        np.testing.assert_almost_equal(self.q1 * self.q2, qt(-30.8,4.4,6.6,8.8), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2 * self.q1, qt(-30.8,4.4,6.6,8.8), decimal=14, err_msg='', verbose=True)
        # Mul int
        np.testing.assert_almost_equal(self.q2 * 2, qt(2.2,4.4,6.6,8.8), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(2 * self.q2, qt(2.2,4.4,6.6,8.8), decimal=14, err_msg='', verbose=True)
        # Mul float
        np.testing.assert_almost_equal(self.q2 * 2.2, qt(2.42,4.84,7.26,9.68), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(2.2 * self.q2, qt(2.42,4.84,7.26,9.68), decimal=14, err_msg='', verbose=True)
        # Mul complex
        np.testing.assert_almost_equal(self.q2 * (1+2j), qt(-3.3,4.4,12.1,-2.2), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal((1+2j) * self.q2, qt(-3.3,4.4,-5.5,11), decimal=14, err_msg='', verbose=True)
        # Small
        np.testing.assert_almost_equal(self.q3 * self.q2, qt(8.426e-15,2.3507e-14,2.8732e-14,4.7091e-14), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2 * self.q3, qt(8.426e-15,2.2693e-14,3.7488e-14,4.0931e-14), decimal=14, err_msg='', verbose=True)
        # Big
        np.testing.assert_almost_equal(self.q4 * self.q2, qt(-4.7509e17,-2.8468e17,2.2693e17,9.174e16), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2 * self.q4, qt(-4.7509e17,2.8732e17,-2.0427e17,1.2914e17), decimal=14, err_msg='', verbose=True)

    def test_conjugate(self):
        self.assertEqual(self.q0.conjugate(), qt(0,0,0,0))
        self.assertEqual(self.q1.conjugate(), qt(1,-2,-3,-4))
        self.assertEqual(self.q2.conjugate(), qt(1.1,-2.2,-3.3,-4.4))
        self.assertEqual(self.q3.conjugate(), qt(1e-14, -1e-15, -1e-16, -1e-17))
        self.assertEqual(self.q4.conjugate(), qt(1e14, -1e15, -1e16, -1e17))

    def test_inverse(self):
        np.testing.assert_almost_equal(self.q1.inverse(), qt(1/30, -1/15,-1/10,-2/15), decimal=14, err_msg='', verbose=True)
        # Decrease dp because my method more precise than Wolfram
        np.testing.assert_almost_equal(self.q2.inverse(), qt(0.030303,-0.0606061,-0.0909091,-0.121212), decimal=6, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q3.inverse(), qt(100000000000000000000/1010101,-10000000000000000000/1010101,-1000000000000000000/1010101,-100000000000000000/1010101), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q4.inverse(), qt(1/101010100000000000000,-1/10101010000000000000,-1/1010101000000000000,-1/101010100000000000), decimal=14, err_msg='', verbose=True)
        
        # Inverse of Zero quaternion raises zero division error
        with self.assertRaises(ZeroDivisionError):
            self.q0.inverse()
        
        # Multiplying Inverses
        np.testing.assert_almost_equal(self.q1 * self.q1.inverse(), qt(1,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q1.inverse() * self.q1, qt(1,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2 * self.q2.inverse(), qt(1,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2.inverse() * self.q2, qt(1,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q3 * self.q3.inverse(), qt(1,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q3.inverse() * self.q3, qt(1,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q4 * self.q4.inverse(), qt(1,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q4.inverse() * self.q4, qt(1,0,0,0), decimal=14, err_msg='', verbose=True)

    def test_div_rdiv(self):
        # Div quaternions
        np.testing.assert_almost_equal(self.q1 / self.q2, qt(0.9090909090909092,-5.551115123125783e-17,2.7755575615628914e-17,4.163336342344337e-17), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2 / self.q1, qt(1.1,1.11022e-16,-2.77556e-17,-5.55112e-17), decimal=14, err_msg='', verbose=True)
        # Div int
        np.testing.assert_almost_equal(self.q2 / 2, qt(0.55,1.1,1.65,2.2), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(2 / self.q2, qt(0.060606060606060615,-0.12121212121212123,-0.18181818181818182,-0.24242424242424246), decimal=14, err_msg='', verbose=True)
        ## Div float
        np.testing.assert_almost_equal(self.q2 / 2.2, qt(0.5,1.0,1.4999999999999998,2.0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(2.2 / self.q2, qt(0.06666666666666668,-0.13333333333333336,-0.2,-0.2666666666666667), decimal=14, err_msg='', verbose=True)
        ## Div complex
        np.testing.assert_almost_equal(self.q2 / (1+2j), qt(1.1,0.0,-1.1,2.2), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal((1+2j) / self.q2, qt(0.15151515151515155,0.0,-0.33333333333333337,0.060606060606060594), decimal=14, err_msg='', verbose=True)
        ## Small
        np.testing.assert_almost_equal(self.q3 / self.q2, qt(3.73939393939394e-16,-5.86969696969697e-16,-7.854545454545455e-16,-1.2966666666666669e-15), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2 / self.q3, qt(134382601343825.98,210939302109392.97,282268802822687.94,465983104659830.94), decimal=14, err_msg='', verbose=True)
        ## Big
        np.testing.assert_almost_equal(self.q4 / self.q2, qt(1.3093939393939372e+16,7903030303030304.0,-5645454545454545.0,3533333333333333.0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2 / self.q4, qt(4.705569047055691e-17,-2.84011202840112e-17,2.0288070202880704e-17,-1.2697740126977403e-17), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q4 / self.q1, qt(1.44033333333333e+16,8693333333333337.0,-6210000000000003.0,3886666666666663.5), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q1 / self.q4, qt(4.277790042777901e-17,2.5819200258192004e-17,1.8443700184437005e-17,-1.1543400115434002e-17), decimal=14, err_msg='', verbose=True)

    def test_eq(self):
        # Test True
        self.assertEqual(self.q0 == qt(0,0,0,0), True)
        self.assertEqual(self.q1 == qt(1,2,3,4), True)
        self.assertEqual(self.q2 == qt(1.1,2.2,3.3,4.4), True)
        self.assertEqual(self.q3 == qt(1e-14, 1e-15, 1e-16, 1e-17), True)
        self.assertEqual(self.q4 == qt(1e14, 1e15, 1e16, 1e17), True)
        # Test False
        self.assertEqual(self.q0 == qt(0,1,0,2000), False)
        self.assertEqual(self.q1 == qt(1,0,3,4), False)
        self.assertEqual(self.q2 == qt(1,2.2,3.3,4.4), False)
        self.assertEqual(self.q3 == qt(1e-14, 1e-15, 1e-16, 1e-19), False)
        self.assertEqual(self.q4 == qt(1e12, 1e15, 1e16, 1e17), False)

    def test_pow(self):
        # Pos n
        np.testing.assert_almost_equal(self.q0**4, qt(0,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q1**4, self.q1*self.q1*self.q1*self.q1, decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2**4, self.q2*self.q2*self.q2*self.q2, decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q3**4, self.q3*self.q3*self.q3*self.q3, decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q4**4, self.q4*self.q4*self.q4*self.q4, decimal=14, err_msg='', verbose=True)
        # Neg n
        np.testing.assert_almost_equal(self.q1**-4, (self.q1.inverse()*self.q1.inverse()*self.q1.inverse()*self.q1.inverse()), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2**-4, (self.q2.inverse()*self.q2.inverse()*self.q2.inverse()*self.q2.inverse()), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q3**-4, (self.q3.inverse()*self.q3.inverse()*self.q3.inverse()*self.q3.inverse()), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q4**-4, (self.q4.inverse()*self.q4.inverse()*self.q4.inverse()*self.q4.inverse()), decimal=14, err_msg='', verbose=True)
        # n == 0
        np.testing.assert_almost_equal(self.q1**0, qt(1,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q2**0, qt(1,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q3**0, qt(1,0,0,0), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.q4**0, qt(1,0,0,0), decimal=14, err_msg='', verbose=True)

        # For qt(0,0,0,0) inverse raises zero division error
        with self.assertRaises(ZeroDivisionError):
            self.q0**-4

    def test_abs(self):
        np.testing.assert_almost_equal(abs(self.q0), 0.0, decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(abs(self.q1), np.sqrt((self.q1.conjugate() * self.q1)[0]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(abs(self.q2), np.sqrt((self.q2.conjugate() * self.q2)[0]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(abs(self.q3), np.sqrt((self.q3.conjugate() * self.q3)[0]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(abs(self.q4), np.sqrt((self.q4.conjugate() * self.q4)[0]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_almost_equal(abs(qt(2, 24, 16, 8)), 30.0, decimal=14, err_msg='', verbose=True)

    def test_nparrays(self):
        q = qt(1,2,3,4)
        r = qt(5,6,7,8)
        t = qt(9,10,11,12)
        s = qt(-3,-7,-4,8)

        A = np.array([[q,r],[s,t]])
        B = np.array([[q,q],[q,q]])
        C = np.array([[self.q2,self.q2],[self.q2,self.q2]])
        D = np.array([[self.q3,self.q3],[self.q3,self.q3]])
        E = np.array([[self.q4,self.q4],[self.q4,self.q4]])

        np.testing.assert_array_almost_equal((A @ A), np.array([[qt(-37,39,-139,49),qt(-248,112,156,152)], [qt(-18,-10,-202,56),qt(-293,39,261,207)]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((B @ B), np.array([[q*q+q*q,q*q+q*q], [q*q+q*q,q*q+q*q]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((C @ C), np.array([[self.q2*self.q2+self.q2*self.q2,self.q2*self.q2+self.q2*self.q2], [self.q2*self.q2+self.q2*self.q2,self.q2*self.q2+self.q2*self.q2]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((D @ D), np.array([[self.q3*self.q3+self.q3*self.q3,self.q3*self.q3+self.q3*self.q3], [self.q3*self.q3+self.q3*self.q3,self.q3*self.q3+self.q3*self.q3]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((E @ E), np.array([[self.q4*self.q4+self.q4*self.q4,self.q4*self.q4+self.q4*self.q4], [self.q4*self.q4+self.q4*self.q4,self.q4*self.q4+self.q4*self.q4]]), decimal=14, err_msg='', verbose=True)
        
        np.testing.assert_array_almost_equal((C @ D), np.array([[self.q2*self.q3+self.q2*self.q3,self.q2*self.q3+self.q2*self.q3], [self.q2*self.q3+self.q2*self.q3,self.q2*self.q3+self.q2*self.q3]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((D @ E), np.array([[self.q3*self.q4+self.q3*self.q4,self.q3*self.q4+self.q3*self.q4], [self.q3*self.q4+self.q3*self.q4,self.q3*self.q4+self.q3*self.q4]]), decimal=14, err_msg='', verbose=True)
        # Matrix mult with big and small quaternions not accurate to 14dp
        #np.testing.assert_array_almost_equal((A @ D) - np.array([[q*self.q3+r*self.q3,s*self.q3+t*self.q3], [q*self.q3+r*self.q3,s*self.q3+t*self.q3]]), np.array([[self.q0,self.q0],[self.q0,self.q0]]), decimal=14, err_msg='', verbose=True)
        #np.testing.assert_array_almost_equal((A @ E) - np.array([[q*self.q4+r*self.q4,s*self.q4+t*self.q4], [q*self.q4+r*self.q4,s*self.q4+t*self.q4]]), np.array([[self.q0,self.q0],[self.q0,self.q0]]), decimal=14, err_msg='', verbose=True)

        # Matrix addition
        np.testing.assert_array_almost_equal((A + A), np.array([[q+q, r+r],[s+s,t+t]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((A + B), np.array([[q+q, r+q],[s+q,t+q]]), decimal=14, err_msg='', verbose=True)
        # Matrix subtraction
        np.testing.assert_array_almost_equal((A - A), np.array([[q-q, r-r],[s-s,t-t]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((A - B), np.array([[q-q, r-q],[s-q,t-q]]), decimal=14, err_msg='', verbose=True)
        # Matrix multiplication
        np.testing.assert_array_almost_equal((A @ A), np.array([[q*q+r*s, q*r+r*t],[s*q+t*s,s*r+t*t]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((A @ B), np.array([[q*q+r*q, q*q+r*q],[s*q+t*q,s*q+t*q]]), decimal=14, err_msg='', verbose=True)
        # Matrix division
        np.testing.assert_array_almost_equal((A / A), np.divide(A, A), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((A / B), np.divide(A, B), decimal=14, err_msg='', verbose=True)

        # Add int, quat matrix
        np.testing.assert_array_almost_equal((1 + A), np.array([[1 + q, 1+r],[1+s,1+t]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((A + 1), np.array([[1 + q, 1+r],[1+s,1+t]]), decimal=14, err_msg='', verbose=True)
        # Sub int, quat matrix
        np.testing.assert_array_almost_equal((A - 1), np.array([[q-1,r-1],[s-1,t-1]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((1 - A), np.array([[1-q,1-r],[1-s,1-t]]), decimal=14, err_msg='', verbose=True)
        # Mul int, quat matrix
        np.testing.assert_array_almost_equal((2 * A), np.array([[2*q,2*r],[2*s,2*t]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((A * 2), np.array([[q*2,r*2],[s*2,t*2]]), decimal=14, err_msg='', verbose=True)
        # Div int, quat matrix
        np.testing.assert_array_almost_equal((A / 2), np.array([[q*1/2,r*1/2],[s*1/2,t*1/2]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((2 / A), np.array([[2*q.inverse(),2*r.inverse()],[2*s.inverse(),2*t.inverse()]]), decimal=14, err_msg='', verbose=True)

        # Add float, quat matrix
        np.testing.assert_array_almost_equal((1.1 + A), np.array([[1.1 + q, 1.1+r],[1.1+s,1.1+t]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((A + 1.1), np.array([[1.1 + q, 1.1+r],[1.1+s,1.1+t]]), decimal=14, err_msg='', verbose=True)
        # Sub float, quat matrix
        np.testing.assert_array_almost_equal((A - 1.1), np.array([[q-1.1,r-1.1],[s-1.1,t-1.1]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((1.1 - A), np.array([[1.1-q,1.1-r],[1.1-s,1.1-t]]), decimal=14, err_msg='', verbose=True)
        # Mul float, quat matrix
        np.testing.assert_array_almost_equal((2.2 * A), np.array([[2.2*q,2.2*r],[2.2*s,2.2*t]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((A * 2.2), np.array([[q*2.2,r*2.2],[s*2.2,t*2.2]]), decimal=14, err_msg='', verbose=True)
        # Div float, quat matrix
        np.testing.assert_array_almost_equal((A / 2.2), np.array([[q*1/2.2,r*1/2.2],[s*1/2.2,t*1/2.2]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((2.2 / A), np.array([[2.2*q.inverse(),2.2*r.inverse()],[2.2*s.inverse(),2.2*t.inverse()]]), decimal=14, err_msg='', verbose=True)

        # Add complex, quat matrix
        np.testing.assert_array_almost_equal(((1+2j) + A), np.array([[(1+2j)+q, (1+2j)+r],[(1+2j)+s,(1+2j)+t]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((A + (1+2j)), np.array([[(1+2j) + q, (1+2j)+r],[(1+2j)+s,(1+2j)+t]]), decimal=14, err_msg='', verbose=True)
        # Sub complex, quat matrix
        np.testing.assert_array_almost_equal((A - (1+2j)), np.array([[q-(1+2j),r-(1+2j)],[s-(1+2j),t-(1+2j)]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal(((1+2j) - A), np.array([[(1+2j)-q,(1+2j)-r],[(1+2j)-s,(1+2j)-t]]), decimal=14, err_msg='', verbose=True)
        # Mul complex, quat matrix
        np.testing.assert_array_almost_equal(((1+2j) * A), np.array([[(1+2j)*q,(1+2j)*r],[(1+2j)*s,(1+2j)*t]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal((A * (1+2j)), np.array([[q*(1+2j),r*(1+2j)],[s*(1+2j),t*(1+2j)]]), decimal=14, err_msg='', verbose=True)
        # Div complex, quat matrix
        np.testing.assert_array_almost_equal((A / (1+2j)), np.array([[q/(1+2j),r/(1+2j)],[s/(1+2j),t/(1+2j)]]), decimal=14, err_msg='', verbose=True)
        np.testing.assert_array_almost_equal(((1+2j) / A), np.array([[(1+2j)/q,(1+2j)/r],[(1+2j)/s,(1+2j)/t]]), decimal=14, err_msg='', verbose=True)

        # Add quat, quat matrix
        # Works when matrix is on the left with quat on right, but not other way around
        # as the addition will belong to the quaternion if we do q + A and this is not implemented.
        np.testing.assert_array_almost_equal((A + q), np.array([[q+q,q+r],[q+s,q+t]]), decimal=14, err_msg='', verbose=True)
        #np.testing.assert_array_almost_equal((q + A), np.array([[q+q,q+r],[q+s,q+t]]), decimal=14, err_msg='', verbose=True)
        # Sub quat, quat matrix
        np.testing.assert_array_almost_equal((A - q), np.array([[q-q,r-q],[s-q,t-q]]), decimal=14, err_msg='', verbose=True)
        #np.testing.assert_array_almost_equal((q - A), np.array([[q-q,q-r],[q-s,q-t]]), decimal=14, err_msg='', verbose=True)
        # Mul quat, quat matrix
        np.testing.assert_array_almost_equal((A * q), np.array([[q*q,r*q],[s*q,t*q]]), decimal=14, err_msg='', verbose=True)
        #np.testing.assert_array_almost_equal((q * A), np.array([[q*q,q*r],[q*s,q*t]]), decimal=14, err_msg='', verbose=True)
        # Div quat, quat matrix
        np.testing.assert_array_almost_equal((A / q), np.array([[q/q,r/q],[s/q,t/q]]), decimal=14, err_msg='', verbose=True)
        #np.testing.assert_array_almost_equal((q / A), np.array([[q/q,q/r],[q/s,q/t]]), decimal=14, err_msg='', verbose=True)


        #self.q1 = qt(1,2,3,4)
        #self.q2 = qt(1.1,2.2,3.3,4.4)
        #self.q3 = qt(1e-14, 1e-15, 1e-16, 1e-17)
        #self.q4 = qt(1e14, 1e15, 1e16, 1e17)

if __name__ == "__main__":
   unittest.main()