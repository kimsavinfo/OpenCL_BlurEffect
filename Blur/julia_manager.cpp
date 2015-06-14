typedef struct cuComplex
{
    float r;
    float i;
    
} cuComplex;

cuComplex createComplex(float p_r, float p_i)
{
    cuComplex complex;
    complex.r = p_r;
    complex.i = p_i;
    return complex;
}

cuComplex multiply(cuComplex p_complex1, cuComplex p_complex2)
{
    return  createComplex(
                          p_complex1.r * p_complex2.r - p_complex1.i * p_complex2.i,
                          p_complex1.i * p_complex2.r + p_complex1.r * p_complex2.i
                          );
    
}

cuComplex add(cuComplex p_complex1, cuComplex p_complex2)
{
    return createComplex(
                         p_complex1.r + p_complex2.r,
                         p_complex1.i + p_complex2.i
                         );
    
}

float magnitude2(cuComplex z)
{
    return z.r * z.r + z.i * z.i;
}

int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(GLOBAL_DIM / 2 - x) / (GLOBAL_DIM / 2);
    float jy = scale * (float)(GLOBAL_DIM / 2 - y) / (GLOBAL_DIM / 2);
    
    cuComplex c = createComplex(-0.8, 0.156);
    cuComplex a = createComplex(jx, jy);
    
    int i = 0;
    for (i = 0; i<200; i++) {
        a = add(multiply(a, a), c);
        if (magnitude2(a) > 1000)
        return 0;
    }
    
    return 1;
}