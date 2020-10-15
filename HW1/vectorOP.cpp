#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N) {
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll);  // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll);  // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative);  //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative);  // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i,
                    maskIsNotNegative);  //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N) {
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    int mask_width = i + VECTOR_WIDTH < N ? VECTOR_WIDTH : N - i;
    __pp_mask activate_mask = _pp_init_ones(mask_width);

    __pp_vec_float x;
    _pp_vload_float(x, values + i, activate_mask);
    __pp_vec_int y;
    _pp_vload_int(y, exponents + i, activate_mask);

    __pp_vec_float out;
    __pp_vec_int one = _pp_vset_int(1);
    __pp_vec_int zero = _pp_vset_int(0);
    __pp_mask y_eq_zero = _pp_init_ones(0);
    _pp_veq_int(y_eq_zero, y, zero, activate_mask);
    __pp_mask tmp = _pp_mask_not(y_eq_zero);
    __pp_mask y_not_zero = _pp_mask_and(activate_mask, tmp);

    __pp_vec_float result;
    __pp_vec_int count;
    _pp_vmove_float(result, x, activate_mask);
    _pp_vmove_int(count, y, activate_mask);

    __pp_mask loop_mask = _pp_init_ones(0);
    _pp_vgt_int(loop_mask, count, one, activate_mask);
    while (_pp_cntbits(loop_mask) != 0) {
      _pp_vmult_float(result, result, x, loop_mask);
      _pp_vsub_int(count, count, one, loop_mask);
      _pp_vgt_int(loop_mask, count, one, loop_mask);
    }
    __pp_mask cap_mask = _pp_init_ones(0);
    __pp_vec_float cap = _pp_vset_float(9.999999f);
    _pp_vgt_float(cap_mask, result, cap, y_not_zero);
    _pp_vmove_float(result, cap, cap_mask);
    __pp_vec_float one_f = _pp_vset_float(1.f);
    _pp_vmove_float(out, one_f, y_eq_zero);
    _pp_vmove_float(out, result, y_not_zero);
    _pp_vstore_float(output + i, out, activate_mask);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N) {
  __pp_mask mask_all = _pp_init_ones();
  __pp_vec_float sum = _pp_vset_float(0);
  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    __pp_vec_float V;
    _pp_vload_float(V, values + i, mask_all);
    _pp_vadd_float(sum, sum, V, mask_all);
  }
  for (int i = VECTOR_WIDTH; i > 1; i /= 2){
    _pp_hadd_float(sum, sum);
    _pp_interleave_float(sum, sum);
  }
  float r[1];
  __pp_mask mask_one = _pp_init_ones(1);
  _pp_vstore_float(r, sum, mask_one);
  return r[0];
}