I have questions regarding step2 and potential parts to modify. For Accum init,
it shouldn't be hardcoded to _mm512_loadu_ps(bias + n + r*16). It should be
the register or variable storing the loaded bias and  _mm512_loadu_ps(bias + n
+ r*16) should be something emitted based on the information from the node for
bias_load = LinearLoad(bias_buf, [n]). It does not have to be stored in a
explicit variable, but at least the staging should be done when visiting the
LinearLoad for the bias and recursively returned so that BinaryMapAccum emits
its arguments in a general way without being hardcoded to a specific condition
on the input arguments.

Does inlining like below make a big difference in performance? For example,
```cpp
// With inlining
acc0 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w_row), acc0);
acc1 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w_row + 16), acc1);
acc2 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w_row + 32), acc2);
acc3 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w_row + 48), acc3);

// without inlining
w_0 = _mm512_loadu_ps(w_row)
w_0 = _mm512_loadu_ps(w_row + 16)
w_0 = _mm512_loadu_ps(w_row + 32)
w_0 = _mm512_loadu_ps(w_row + 48)
acc0 = _mm512_fmadd_ps(xk, w_0, acc0);
acc1 = _mm512_fmadd_ps(xk, w_1, acc1);
acc2 = _mm512_fmadd_ps(xk, w_2, acc2);
acc3 = _mm512_fmadd_ps(xk, w_3, acc3);
```