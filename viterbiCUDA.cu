#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>
#include <fstream>

using namespace std;
#define MAX_T 10000
#define MIN_T 1000
#define MAX_K 400
#define MIN_K  50
#define N 10

#define NUM_SAMPLES 50

double T1[MAX_K][MAX_T];
int T2[MAX_K][MAX_T];
double pi[MAX_K];
int Y[MAX_T];
int X[MAX_T];
double A[MAX_K][MAX_K];
double B[MAX_K][N];


void printPath(int nT){

    for (int i = 0; i < nT-1; ++i) {

        cout << X[i] << " -> ";
    }
    cout << X[nT-1] <<"\n";
}

void generateTransitionProbabilities(){

    double sum;
    double rands[MAX_K];

    for(int i =0; i<MAX_K;i++){

        sum = 0.0;
            for (int l = 0; l < MAX_K; ++l) {

                rands[l] = (double)rand();
                sum += rands[l];

            }



        for (int j = 0; j < MAX_K; j++) {


            A[i][j] = rands[j] / sum;
        }





    }


}

void generateEmmisionProbabilities(){

    double sum;
    double rands[N];

    for(int i =0; i<MAX_K;i++){

        sum = 0.0;
        for (int l = 0; l < N; ++l) {

            rands[l] = (double)rand();
            sum += rands[l];

        }



        for (int j = 0; j < N; j++) {


            B[i][j] = rands[j] / sum;
        }





    }


}

void generatePriors(){

    double rands[MAX_K];
    double sum = 0.0;
    for (int i = 0; i < MAX_K; ++i) {

        rands[i] = (double)rand();
        sum += rands[i];

    }

    for (int j = 0; j <MAX_K; ++j) {
        pi[j] = rands[j]/sum;
    }


}

void generateObservations(){


    random_device rd;
    mt19937 eng(rd());
    uniform_int_distribution<> distr(0, N-1);

    for(int i=0; i<MAX_T; ++i)
        Y[i] = distr(eng);




}

void generateInputs(){
    srand(time(NULL)); // New random seed

    generateTransitionProbabilities();
    generateEmmisionProbabilities();
    generatePriors();

    generateObservations();
}



void viterbiCPU(int nK, int nT){

    double maxT1=-1;
    double maxT2=-1;
    double tempMaxT1;
    double tempMaxT2;




    int argmax=-1;
    for (int i = 0; i <nK; ++i) { // For each state

        T1[i][0] = pi[i]*B[i][Y[0]];
        T2[i][0] = 0.0;
    }


    for (int j = 1; j <nT ; ++j) {  // For each observation

        for (int i = 0; i < nK; ++i) { // For each state

            maxT1 = -1;
            maxT2 = -1;
            for (int k = 0; k < nK; ++k) { // For each state

                tempMaxT2 = T1[k][j-1] * A[k][i];

                tempMaxT1= tempMaxT2*B[i][Y[j]];

                if( tempMaxT1 > maxT1) maxT1=tempMaxT1;

                if( tempMaxT2 > maxT2){

                    maxT2 = tempMaxT2;
                    argmax = k;
                }

            }

            T1[i][j] = maxT1;
            T2[i][j] = argmax;

        }
    }
    maxT1 = -1;
    argmax = -1;
    for (int k = 0; k < nK; ++k) {

        if(T1[k][nT-1] > maxT1){
            maxT1 = T1[k][nT-1];
            argmax = k;
        }

    }

    X[nT-1] = argmax;

    for (int j = nT-1; j >0 ; j--) {
        //
        X[j-1] = T2[X[j]][j];
    }


    }

double cpuFLOPS(double time, float k, float t){

    return (3*k +(3*k*k + 1)*(t-1))/(time + 1e-16);


}

__global__ void viterbi(int t, double *t1, int *t2, double * pi, int * y, int *x, double *a, double * b ){

  int idx = threadIdx.x;

  int states = blockDim.x;

  double tempTA;
  double maxTA=-1.0;
  int argmax;
  double tempTAB;
  double maxTAB =-1.0;
  double maxT1End = -1.0;
  double tempMaxT1End;
  int argmaxT1End;
  // Index t2 t2[i*MAX_T+j];
  // Index t1 t1[i*MAX_T+j];


  for(int j = 0; j <t;j++){

    if(j==0){

      t1[idx*MAX_T] = pi[idx]*b[idx*MAX_K+y[0]];

    }else{

      maxTA = -1.0;
      maxTAB = -1.0;

      for(int k=0; k< states; k++){

        tempTA = t1[k*MAX_T +j-1]* a[k*MAX_K+idx];

        if(tempTA>maxTA){

            maxTA = tempTA;
            argmax = k;
        }

        tempTAB = tempTA* b[idx*MAX_K+y[j]];

        if(tempTAB>maxTAB) maxTAB=tempTAB;

      }
      t1[idx*MAX_T+j] = maxTAB;
      t2[idx*MAX_T+j] = argmax;



    }


    __syncthreads();
  }


    if(idx==0){

      for(int k =0; k <states; k++){

      tempMaxT1End = t1[k*MAX_T+t-1];

      if(tempMaxT1End>maxT1End){

        maxT1End = tempMaxT1End;
        argmaxT1End = k;

      }

      }
      x[t-1] = argmaxT1End;
      for(int j = t-1; j>0; j--) x[j-1]= t2[x[j]*MAX_T+j];

      // for (int j = 0; j < 10; j++) {
      //
      //     printf("%d ->", x[j]);
      // }
      // printf("\n");
    }




}

double GPUFlops(double time, int k, int t){

  return (double)(2*k*k*(t-1) + 2*k) / (time);

}

int main() {
    generateInputs();

    double *d_T1;
    int *d_T2;
    double *d_pi;
    int *d_Y;
    int *d_X;
    double *d_A;
    double *d_B;

    int T1_size = MAX_K*MAX_T * sizeof(double);
    int T2_size = MAX_K*MAX_T * sizeof(int);
    int pi_size = MAX_K*sizeof(double);
    int Y_size = MAX_T * sizeof(int);
    int X_size = MAX_T * sizeof(int);
    int A_size = MAX_K * MAX_K * sizeof(double);
    int B_size = MAX_K * N * sizeof(double);

    cudaMalloc((void**)&d_T1, T1_size);
    cudaMalloc((void**)&d_T2, T2_size);
    cudaMalloc((void**)&d_pi, pi_size);
    cudaMalloc((void**)&d_Y, Y_size);
    cudaMalloc((void**)&d_X, X_size);
    cudaMalloc((void**)&d_A, A_size);
    cudaMalloc((void**)&d_B, B_size);

    cudaMemcpy(d_X,X, X_size, cudaMemcpyHostToDevice);
    int k = MAX_K;
    int t = MAX_K;


    cudaMemcpy(d_T1,T1, T1_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_T2,T2, T2_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_pi,pi, pi_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_Y,Y, Y_size, cudaMemcpyHostToDevice);



    cudaMemcpy(d_A,A, A_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_B,B, B_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    //
    //
    // clock_t start = clock();
    // viterbi<<<1,k>>>(t, d_T1, d_T2, d_pi, d_Y, d_X,d_A,d_B); //Kernel invocation
    // cudaDeviceSynchronize();
    // clock_t end = clock();
    //
    // viterbiCPU(k,t);
    // printPath(10);

    int Ks[NUM_SAMPLES];
    int Ts[NUM_SAMPLES];
    double Ktimes[NUM_SAMPLES], Ttimes[NUM_SAMPLES],Kflops[NUM_SAMPLES], Tflops[NUM_SAMPLES];


    for (int i = 1; i < NUM_SAMPLES+1; ++i) {

        Ks[i-1] =MIN_K +  (i-1)*((MAX_K-MIN_K)/NUM_SAMPLES);
        Ts[i-1] =MIN_T + (i-1)*((MAX_T-MIN_T)/NUM_SAMPLES);
    }

for(int i =0; i < NUM_SAMPLES; i++){
k = Ks[i];
t = MAX_T;

clock_t start = clock();
viterbi<<<1,k>>>(t, d_T1, d_T2, d_pi, d_Y, d_X,d_A,d_B); //Kernel invocation
cudaDeviceSynchronize();
clock_t end = clock();


Ktimes[i] = (end - start)/ (double) CLOCKS_PER_SEC;

Kflops[i] = GPUFlops(Ktimes[i], k,t);

}
for(int i =0; i < NUM_SAMPLES; i++){
k = MAX_K;
t = Ts[i];

clock_t start = clock();
viterbi<<<1,k>>>(t, d_T1, d_T2, d_pi, d_Y, d_X,d_A,d_B); //Kernel invocation
cudaDeviceSynchronize();
clock_t end = clock();


Ttimes[i] = (end - start)/ (double) CLOCKS_PER_SEC;

Tflops[i] = GPUFlops(Ttimes[i], k,t);
}


ofstream outK("CUDAvariedK.txt");
ofstream outT("CUDAvariedT.txt");


outK << "K, T, Time, Flops\n";
outT << "K, T, Time, Flops\n";

for (int k = 0; k <NUM_SAMPLES ; ++k) {
    outK << Ks[k] << ", " << MAX_T << ", " << Ktimes[k] << ", " << Kflops[k] <<endl;
    outT << MAX_K << ", " << Ts[k] << ", " << Ttimes[k] << ", " << Tflops[k] <<endl;

}
    return 0;



}
