using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ComputationalExamples
{
    class Program
    {
        public static Random rnd = new Random();

        public static List<float> ExpectedResults = new List<float>();
        public static List<float> ActualResults = new List<float>();

        static void Evaluate(string Name)
        {
            Console.WriteLine("Evaluating {0} function...", Name);
            int failed = 0;
            for (int i=0; i<ExpectedResults.Count; i++)
            {
                if (ExpectedResults[i] != ActualResults[i])
                {
                    Console.WriteLine("Error: Test Case {0}, Expected: {1}, Actual: {2}", i, ExpectedResults[i], ActualResults[i]);
                    failed++;
                }
            }
            Console.WriteLine("{0} test failed {1} tests of {2}", Name, failed, ExpectedResults.Count);
        }

        static void Main(string[] args)
        {
            var device = CNTK.DeviceDescriptor.CPUDevice;

            Not.RunTest(1000, device); Evaluate("Not");
            And.RunTest(1000, device); Evaluate("And");
            mAdd.RunTest(1000, device); Evaluate("Multiply-Add");
        }
    }
}
