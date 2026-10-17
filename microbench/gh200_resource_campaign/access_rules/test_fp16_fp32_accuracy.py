import unittest
import numpy as np
import fp16_fp32_accuracy as accuracy


class AccuracyBound(unittest.TestCase):
    def inputs(self,k=32768):
        rows=np.arange(32,dtype=np.uint32)[:,None]
        inner=np.arange(k,dtype=np.uint32)[None,:]
        return (accuracy.random_values(rows,inner,29,True),
                accuracy.random_values(inner,rows+17,29,False))

    def test_rounding_orders(self):
        a,b=self.inputs();expected,bound=accuracy.dot_reference_bound(a,b)
        for rounding in ('rn','rz'):
            for reverse in (False,True):
                acc=np.zeros(len(a),dtype=np.float32)
                for begin in range(0,a.shape[1],16):
                    order=range(begin,begin+16)
                    if reverse:order=reversed(order)
                    for index in order:
                        exact=acc.astype(np.float64)+a[:,index]*b[:,index]
                        value=exact.astype(np.float32)
                        if rounding=='rz':
                            value=np.where(np.abs(value.astype(np.float64))>np.abs(exact),np.nextafter(value,np.float32(0)),value)
                        acc=value
                self.assertTrue(np.all(np.abs(acc-expected)<=bound),(rounding,reverse))

    def test_subnormal_flush(self):
        a=np.full((2,64),2**-20);b=np.ones_like(a)
        expected,bound=accuracy.dot_reference_bound(a,b)
        self.assertTrue(np.all(np.abs(expected)<=bound))

    def test_omitted_tiles_detected(self):
        a,b=self.inputs();expected,bound=accuracy.dot_reference_bound(a,b)
        products=a*b
        for start in (0,16384,32704):
            wrong=expected-products[:,start:start+64].sum(axis=1)
            self.assertTrue(np.any(np.abs(wrong-expected)>bound),start)

    def test_wrong_input_index_detected(self):
        a,b=self.inputs();expected,bound=accuracy.dot_reference_bound(a,b)
        wrong=(np.roll(a,1,axis=0)*b).sum(axis=1)
        self.assertTrue(np.all(np.abs(wrong-expected)>bound))

    def test_scalar_recurrence(self):
        a=np.array([[1.,-1.]*16]);b=np.ones_like(a)
        expected,bound=accuracy.dot_reference_bound(a,b)
        g=accuracy.GAMMA
        self.assertEqual(expected[0],0)
        self.assertAlmostEqual(bound[0],accuracy.ATOL+(1+g)*(g*16)+g*16)


if __name__=='__main__':unittest.main()
