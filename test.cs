using Microsoft.ML.OnnxRuntime.Tensors;

namespace IntexAPI
{
    public class test1
    {
        public float squarenorthsouth { get; set; }
        public float depth { get; set; }
        public float squareeastwest { get; set; }
        public float length { get; set; }
        public float headdirection_E { get; set; }
        public float headdirection_W { get; set; }

        public float sex_F { get; set; }
        public float sex_M { get; set; }
        public float eastwest_E { get; set; }
        public float eastwest_W {get; set; }
        public float facebundles_N { get; set; }
        public float facebundles_Y { get; set; }
        public float ageatdeath_A { get; set; }
        public float ageatdeath_C { get; set; }
        public float ageatdeath_I { get; set; }
        public float ageatdeath_N { get; set; }

        public Tensor<float> AsTensor()
        {
            float[] data = new float[]
            {
            squarenorthsouth, depth, squareeastwest, length, headdirection_E, headdirection_W, sex_F, sex_M, eastwest_E, eastwest_W, facebundles_N, facebundles_Y, ageatdeath_A, ageatdeath_C, ageatdeath_I, ageatdeath_N
            };
            int[] dimensions = new int[] { 1, 16 };
            return new DenseTensor<float>(data, dimensions);
        }
    }
}


