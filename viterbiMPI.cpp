
#include <mpi.h>
#include <stdio.h>


#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>
#include <fstream>
#include <tuple>
using namespace std;
#define MAX_T 10000
#define MIN_T 1000
#define MAX_K 250
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
    std::cout << X[nT-1] <<"\n";
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

		cout << "\n";

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

void CPUexperiments(bool suppressTracking, bool suppressDetails){

    int Ks[NUM_SAMPLES];
    int Ts[NUM_SAMPLES];
    clock_t startCPU;
    clock_t endCPU;
    double Ktimes[NUM_SAMPLES], Ttimes[NUM_SAMPLES],Kflops[NUM_SAMPLES], Tflops[NUM_SAMPLES];
    double time,flops;


    for (int i = 1; i < NUM_SAMPLES+1; ++i) {

        Ks[i-1] =MIN_K +  (i-1)*((MAX_K-MIN_K)/NUM_SAMPLES);
        Ts[i-1] =MIN_T + (i-1)*((MAX_T-MIN_T)/NUM_SAMPLES);
    }

    if(!suppressTracking) cout << "Starting K experiment\n";

    for (int j = 0; j < NUM_SAMPLES; ++j) {

        if(!suppressDetails){

            cout << "Starting Experiment \t " << j << " of " << NUM_SAMPLES << "\n";
            cout << "Parameters: \t T: " << MAX_T << " \t K: " << Ks[j] << "\n";
        }
        generateInputs(); // Generate new experimental data each experiment

        startCPU = clock();
        viterbiCPU(Ks[j],MAX_T);
        endCPU = clock();

        time = (endCPU - startCPU)/ (double) CLOCKS_PER_SEC;
        flops = cpuFLOPS(time,Ks[j],MAX_T);



        if(!suppressDetails){


            cout << "Completed\n Time: "<<time << " \t FLOPS: " << flops << "\n";
        }
        Ktimes[j] = time;
        Kflops[j] = flops;
    }

    if(!suppressTracking) cout << "Starting T experiment\n";
    for (int j = 0; j < NUM_SAMPLES; ++j) {

        if(!suppressDetails){

            cout << "Starting Experiment \t " << j << " of " << NUM_SAMPLES << "\n";
            cout << "Parameters: \t T: " << Ts[j] << " \t K: " << MAX_K << "\n";
        }
        generateInputs(); // Generate new experimental data each experiment

        startCPU = clock();
        viterbiCPU(MAX_K,Ts[j]);
        endCPU = clock();

        time = (endCPU - startCPU)/ (double) CLOCKS_PER_SEC;
        flops = cpuFLOPS(time,MAX_K,Ts[j]);


        if(!suppressDetails){


            cout << "Completed\n Time: "<<time << " \t FLOPS: " << flops << "\n";
        }

        Ttimes[j] = time;
        Tflops[j] = flops;
    }

    if(!suppressTracking) cout << "Writing Results to file\n";

    ofstream outK("CPUvariedK.txt");
    ofstream outT("CPUvariedT.txt");


    outK << "K, T, Time, Flops\n";
    outT << "K, T, Time, Flops\n";

    for (int k = 0; k <NUM_SAMPLES ; ++k) {
        outK << Ks[k] << ", " << MAX_T << ", " << Ktimes[k] << ", " << Kflops[k] <<endl;
        outT << MAX_K << ", " << Ts[k] << ", " << Ttimes[k] << ", " << Tflops[k] <<endl;

    }

    if(!suppressTracking) cout << "Finished Write\n";

}


tuple<int,int,int>* assignRows(int rank, int k, int procs){

  int * startKs = (int*)malloc(procs*sizeof(int));
  int * endKs = (int*)malloc(procs*sizeof(int));
  int * section_sizes = (int*)malloc(procs*sizeof(int));

  tuple<int,int,int> * assignments = (tuple<int,int,int>*)malloc(procs*sizeof(tuple<int,int,int>));

  for (int i = 0; i < procs; i++) {

    section_sizes[i] = 0;
    startKs[i] = 0;
    endKs[i] = 0;
    }

  for (int i = 0; i < k; i++) {


    section_sizes[i%procs] +=1;

  }

  for(int i =1; i <procs; i++){

    startKs[i] = startKs[i-1]+section_sizes[i-1];


    }

  for (int i = 0; i < procs; i++) {

    endKs[i] = startKs[i] + section_sizes[i] - 1;

  }

for(int i=0;i< procs; i++){


  assignments[i] = make_tuple(startKs[i], endKs[i], section_sizes[i]);

}

  return assignments;

}
//////////////////////////////////////////////////////

void viterbiMPI(int k, int t){


	int num_procs, myrank, section_size, startk, endk, counter;

	generateInputs();

	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	MPI_Datatype y;
	MPI_Type_contiguous(MAX_T, MPI_INT, &y);
	MPI_Type_commit(&y);
	MPI_Bcast(Y, 1,y,0,MPI_COMM_WORLD);

  tuple<int,int,int> * assignments = (tuple<int,int,int>*)malloc(num_procs*sizeof(tuple<int,int,int>));

  assignments = assignRows(myrank,k,num_procs);





  std::tie(startk,endk,section_size) = assignments[myrank];




	double * sendCol = (double*)malloc(section_size*sizeof(double));
	double * recvCol = (double*)malloc(section_size*sizeof(double));

	int * sendColInt = (int*)malloc(section_size*sizeof(int));
	int * recvColInt = (int*)malloc(section_size*sizeof(int));


	double * t1Column = (double*)malloc(k*sizeof(double));
	int * t2Column = (int*)malloc(k*sizeof(int));



	MPI_Datatype colType, colTypeInt;
	MPI_Type_contiguous(section_size, MPI_DOUBLE, &colType);
	MPI_Type_commit(&colType);
	MPI_Type_contiguous(section_size, MPI_DOUBLE, &colTypeInt);
	MPI_Type_commit(&colTypeInt);


  	MPI_Datatype T1Col;
  	MPI_Type_contiguous(k, MPI_DOUBLE, &T1Col);
  	MPI_Type_commit(&T1Col);


    	MPI_Datatype T2Col;
    	MPI_Type_contiguous(k, MPI_INT, &T2Col);
    	MPI_Type_commit(&T2Col);





	for(int i = startk; i <=endk;i++){

		T1[i][0] = pi[i]*B[i][Y[0]];



		sendCol[i-startk] = T1[i][0];

		T2[i][0] = 0;

				}


	MPI_Barrier(MPI_COMM_WORLD);


	/////////////////////////////////////////////////////////////////////////////////////////////////////////


	MPI_Status status;
	int sender;

	if(myrank != 0){

		MPI_Send(sendCol, 1, colType, 0, myrank,MPI_COMM_WORLD);

		}

	else{
    counter = 0;
		while(counter < num_procs-1){

			MPI_Recv(recvCol, 1, colType, MPI_ANY_SOURCE, MPI_ANY_TAG,MPI_COMM_WORLD, &status);

			sender = status.MPI_SOURCE;

			counter++;

      int senderStartK, senderEndK, senderSectionSize;

      std::tie(senderStartK,senderEndK, senderSectionSize) = assignments[sender];




			for(int i = senderStartK; i <= senderEndK; i++){ // populating column from received subcolumns

				T1[i][0] = recvCol[i-senderStartK];


				t1Column[i] = T1[i][0];
				}


			}


		for(int i =startk; i <=endk; i++){

			t1Column[i] = sendCol[i]; // Add node zero's data to column

			}


	}


	MPI_Bcast(t1Column, 1,T1Col,0,MPI_COMM_WORLD); // send the constructed column to all nodes

	for(int i = 0; i<k; i++){

	T1[i][0] = t1Column[i]; // Populate T1 first column for all nodes

	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



		MPI_Barrier(MPI_COMM_WORLD);
	int argmaxTA;
	double maxTA = -1.0;
	double tempTA;
	double maxTAB = -1.0;
	double tempTAB;

	for(int j = 1; j < t; j++){ // For each observation

			//if(myrank == 0) printf("For observation %d\n", j);

		for(int i = startk; i <=endk;i++){ // For each state


			for(int m = 0; m<k; m++){

				tempTA = T1[m][j-1]*A[m][i]; // T1[k,j-1]*A_ki



				if(tempTA >maxTA){

					maxTA = tempTA;
					argmaxTA = m;

					}

				tempTAB = tempTA*B[i][Y[j]];


				if(tempTAB > maxTAB) maxTAB = tempTAB;


				}

			sendCol[i-startk] = maxTAB;
			sendColInt[i-startk] = argmaxTA;

		}

		MPI_Barrier(MPI_COMM_WORLD);

	/////////////////////////////////////////////////////////////////////////////
    if(myrank != 0){


			MPI_Send(sendCol, 1, colType, 0, myrank,MPI_COMM_WORLD);

		}
		else{

			counter = 0;

			while(counter < num_procs-1){

				MPI_Recv(recvCol, 1, colType, MPI_ANY_SOURCE, MPI_ANY_TAG,MPI_COMM_WORLD, &status);



				sender = status.MPI_SOURCE;

				counter++;

        int senderStartK, senderEndK, senderSectionSize;

        std::tie(senderStartK,senderEndK, senderSectionSize) = assignments[sender];




				for(int i = senderStartK; i <= senderEndK; i++){

					T1[i][j] = recvCol[i-senderStartK]; // Populate


					t1Column[i] = T1[i][j]; // Populate column with received subcolumns
					}

			}

			for(int i =startk; i <=endk; i++){

				t1Column[i] = sendCol[i]; // Add local data to column

				}
        //
        //   printf("T1 before: ");
        // for (int i = 0; i < 5; i++) {
        //
        //   printf("%f , ", t1Column[i]);
        //
        // }
        // cout<<endl;
		}

	///////////////////////////////////////////////////////////////////////////////

		MPI_Bcast(t1Column, 1,T1Col,0,MPI_COMM_WORLD);

		for(int i = 0; i<k; i++){

		T1[i][j] = t1Column[i];

		}


		if(myrank != 0){

			MPI_Send(sendColInt, 1, colTypeInt, 0, myrank,MPI_COMM_WORLD);

		}
		else{

			counter = 0;

			while(counter < num_procs-1){

				MPI_Recv(recvColInt, 1, colTypeInt, MPI_ANY_SOURCE, MPI_ANY_TAG,MPI_COMM_WORLD, &status);



				sender = status.MPI_SOURCE;



				counter++;
        int senderStartK, senderEndK, senderSectionSize;

        std::tie(senderStartK,senderEndK, senderSectionSize) = assignments[sender];




				for(int i = senderStartK; i <= senderEndK; i++){

					T2[i][j] = recvColInt[i-senderStartK];


					t2Column[i] = T2[i][j];
					}

			}

			for(int i =startk; i <=endk; i++){

				t2Column[i] = sendColInt[i];

				}


                //   printf("T2 before: ");
                // for (int i = 0; i < 5; i++) {
                //
                //   printf("%d , ", t2Column[i]);
                //
                // }
                // cout<<endl;

		}

	/////////////////////////////////////////////////////

  MPI_Bcast(t2Column, 1,T2Col,0,MPI_COMM_WORLD);


    for (int i = 0; i < k; i++) {

      T2[i][j] = t2Column[i];


  } // Correct

  MPI_Barrier(MPI_COMM_WORLD);

// if(myrank ==0){
//
//   cout<< "For observation " << j << endl;
//
//   for (int i = 0; i < k; i++) {
//     for (int m = 0; m < t; m++) {
//
//       cout << T2[i][m] << " ";
//
//     }
//     cout<<endl;
//   }
//
//   for (int i = 0; i < k; i++) {
//     for (int m = 0; m < t; m++) {
//
//       cout << T1[i][m] << " ";
//
//     }
//     cout<<endl;
//   }
//
// }
// if(myrank == 0) cout << "Completed \n";
// MPI_Barrier(MPI_COMM_WORLD);
// }




}


	if(myrank ==0){

		double max = -1;
		int argmax = -1;
		for(int m = 0; m<k; m++){

			if(T1[m][t-1] > max){

			max = T1[m][t-1];
			argmax = m;

			}

		}

		X[t-1] = argmax;



		for(int j = t-1; j>0; j--) {

			//cout << j << " " << X[j] << "\n";
			X[j-1] = T2[X[j]][j];

		}



	}


}

int main( int argc, char *argv[]){

int k,t;
MPI_Init(&argc,&argv);
int myrank;


MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
// if(myrank ==0){
//
//   k = (int)atoi(argv[1]);
//   t = (int)atoi(argv[2]);
//   printf("%d %d\n", k,t);
// }
// MPI_Bcast(&k, 1,MPI_INT,0,MPI_COMM_WORLD);
// MPI_Bcast(&t, 1,MPI_INT,0,MPI_COMM_WORLD);
//
// if(myrank ==1){
//
// printf("%d %d\n", k,t);
//
// }

generateInputs();
double s = MPI_Wtime();
viterbiMPI(250,10000);
double e = MPI_Wtime();

if(myrank==0) cout << (e-s)/(double)1000.0;

MPI_Finalize();



return 0;

}
