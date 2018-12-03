#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>

float g(double z) {
  float e = 2.71828;

  return 1/(1 + std::pow(e,z));
}

float sigmoidalGradiente(double z) {
  float e = 2.71828;
  float g_z = g(z);

  return g_z * (1 - g_z);
  // return std::pow(e,z)/(std::pow(1+std::pow(e,z),2));
}

class NeuralNetwork {
  private:
    int input_layer_size = 400;
    int hidden_layer_size = 25;
    int output_layer_size = 10;

    std::vector<std::vector<double>> xv;

    std::vector<std::vector<double>> W1;
    std::vector<std::vector<double>> W2;
    std::vector<std::vector<double>> transW2;
    std::vector<std::vector<double>> z2;
    std::vector<std::vector<double>> a2;
    std::vector<std::vector<double>> transa2;
    std::vector<std::vector<double>> delta2;
    std::vector<std::vector<double>> z3;

    std::vector<std::vector<double>> delta3;
    std::vector<std::vector<double>> dJdW1;
    std::vector<std::vector<double>> dJdW2;

  public:
    std::vector<std::vector<double>> yhat;
    void entrenaRN2(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y) {
      double val = 0;
      std::vector<std::vector<double>> transx;
      std::vector<double> row;

      for(int i=0; i<4000; i++) {
    		for(int j=0; j<25; j++) {
    			for(int k=0; k<400; k++) {
            val += x[i][k] * W1[k][j];
    				z2[i][j] += val;
    			}
          // row.push_back(val);
          // val = 0;
    		}
        // z2.push_back(row);
        // row.clear();
    	}

      for(int i=0; i<4000; i++) {
    		for(int j=0; j<25; j++) {
          // row.push_back(g(z2[i][j]));
          a2[i][j] = g(z2[i][j]);
        }
        // a2.push_back(row);
        // row.clear();
      }

      for(int i=0; i<4000; i++) {
    		for(int j=0; j<10; j++) {
    			for(int k=0; k<25; k++) {
            // val += a2[i][k] * W2[k][j];
    				z3[i][j] += a2[i][k] * W2[k][j];
    			}
          // row.push_back(val);
          // val = 0;
    		}
        // z3.push_back(row);
        // row.clear();
    	}

      for(int i=0; i<4000; i++) {
    		for(int j=0; j<10; j++) {
          // row.push_back(g(z3[i][j]));
          yhat[i][j] = g(z3[i][j]);
        }
        // yhat.push_back(row);
        // row.clear();
      }

      for(int i=0; i<4000; i++) {
        for(int j=0; j<10; j++) {
          // row.push_back(-(y[i][j]-yhat[i][j]) * sigmoidalGradiente(z3[i][j]));
          delta3[i][j] = -(y[i][j]-yhat[i][j]) * sigmoidalGradiente(z3[i][j]);
        }
        // delta3.push_back(row);
        // row.clear();
      }

      transa2.resize(25);
      for (int i = 0; i < 25; ++i)
          transa2[i].resize(4000);

      for(int i=0; i<4000; i++) {
        for(int j=0; j<25; j++) {
          transa2[j][i]=a2[i][j];
        }
      }

      for(int i=0; i<25; i++) {
    		for(int j=0; j<10; j++) {
    			for(int k=0; k<4000; k++) {
            // val += transa2[i][k] * delta3[k][j];
    				dJdW2[i][j] += transa2[i][k] * delta3[k][j];
    			}
          row.push_back(val);
          val = 0;
    		}
        dJdW2.push_back(row);
        row.clear();
    	}

      transx.resize(400);
      for (int i = 0; i < 400; ++i)
          transx[i].resize(4000);

      for(int i=0; i<4000; i++) {
        for(int j=0; j<400; j++) {
          transx[j][i] = x[i][j];
        }
      }

      transW2.resize(10);
      for (int i = 0; i < 10; ++i)
          transW2[i].resize(25);

      for(int i=0; i<25; i++) {
        for(int j=0; j<10; j++) {
          // row.push_back(W2[i][j]);
          transW2[j][i] = W2[i][j];
        }
        // transW2.push_back(row);
        // row.clear();
      }

      for(int i=0; i<4000; i++) {
    		for(int j=0; j<25; j++) {
    			for(int k=0; k<10; k++) {
            // val += delta3[i][k] * transW2[k][j];
    				delta2[i][j] += delta3[i][k] * transW2[k][j];
    			}
          // row.push_back(val);
          // val = 0;
    		}
        // delta2.push_back(row);
        // row.clear();
    	}

      for(int i=0; i<400; i++) {
        for(int j=0; j<25; j++) {
          delta2[i][j] = delta2[i][j] * sigmoidalGradiente(z2[i][j]);
        }
      }

      for(int i=0; i<400; i++) {
    		for(int j=0; j<25; j++) {
    			for(int k=0; k<4000; k++) {
            // val += transx[i][k] * delta2[k][j];
    				dJdW1[i][j] += transx[i][k] * delta2[k][j];
    			}
          // row.push_back(val);
          // val = 0;
    		}
        // dJdW1.push_back(row);
        // row.clear();
    	}
    }

    void entrenaRN(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y) {
      double val = 0;
      std::vector<std::vector<double>> transx;
      std::vector<double> row;

      for(int i=0; i<4000; i++) {
    		for(int j=0; j<25; j++) {
    			for(int k=0; k<400; k++) {
            val += x[i][k] * W1[k][j];
    				// z2[i][j] += x[i][k] * W1[k][j];
    			}
          row.push_back(val);
          val = 0;
    		}
        z2.push_back(row);
        row.clear();
    	}

      for(int i=0; i<4000; i++) {
    		for(int j=0; j<25; j++) {
          row.push_back(g(z2[i][j]));
          // a2[i][j] = g(z2[i][j]);
        }
        a2.push_back(row);
        row.clear();
      }

      for(int i=0; i<4000; i++) {
    		for(int j=0; j<10; j++) {
    			for(int k=0; k<25; k++) {
            val += a2[i][k] * W2[k][j];
    				// z3[i][j] += a2[i][k] * W2[k][j];
    			}
          row.push_back(val);
          val = 0;
    		}
        z3.push_back(row);
        row.clear();
    	}

      for(int i=0; i<4000; i++) {
    		for(int j=0; j<10; j++) {
          row.push_back(g(z3[i][j]));
        }
        yhat.push_back(row);
        row.clear();
      }

      for(int i=0; i<4000; i++) {
        for(int j=0; j<10; j++) {
          row.push_back(-(y[i][j]-yhat[i][j]) * sigmoidalGradiente(z3[i][j]));
          // delta3[i][j] = -(y[i]-yhat[i][j]) * sigmoidalGradiente(z3[i][j]);
        }
        delta3.push_back(row);
        row.clear();
      }

      transa2.resize(25);
      for (int i = 0; i < 25; ++i)
          transa2[i].resize(4000);

      for(int i=0; i<4000; i++) {
        for(int j=0; j<25; j++) {
          transa2[j][i]=a2[i][j];
        }
      }

      for(int i=0; i<25; i++) {
    		for(int j=0; j<10; j++) {
    			for(int k=0; k<4000; k++) {
            val += transa2[i][k] * delta3[k][j];
    				// dJdW2[i][j] += transa2[i][k] * delta3[k][j];
    			}
          row.push_back(val);
          val = 0;
    		}
        dJdW2.push_back(row);
        row.clear();
    	}

      transx.resize(400);
      for (int i = 0; i < 400; ++i)
          transx[i].resize(4000);

      for(int i=0; i<4000; i++) {
        for(int j=0; j<400; j++) {
          transx[j][i] = x[i][j];
        }
      }

      transW2.resize(10);
      for (int i = 0; i < 10; ++i)
          transW2[i].resize(25);

      for(int i=0; i<25; i++) {
        for(int j=0; j<10; j++) {
          // row.push_back(W2[i][j]);
          transW2[j][i] = W2[i][j];
        }
        // transW2.push_back(row);
        // row.clear();
      }

      for(int i=0; i<4000; i++) {
    		for(int j=0; j<25; j++) {
    			for(int k=0; k<10; k++) {
            val += delta3[i][k] * transW2[k][j];
    				// delta2[i][j] += transx[i][k] * delta2[k][j];
    			}
          row.push_back(val);
          val = 0;
    		}
        delta2.push_back(row);
        row.clear();
    	}

      for(int i=0; i<400; i++) {
        for(int j=0; j<25; j++) {
          delta2[i][j] = delta2[i][j] * sigmoidalGradiente(z2[i][j]);
        }
      }

      for(int i=0; i<400; i++) {
    		for(int j=0; j<25; j++) {
    			for(int k=0; k<4000; k++) {
            val += transx[i][k] * delta2[k][j];
    				// dJdW1[i][j] += transx[i][k] * delta2[k][j];
    			}
          row.push_back(val);
          val = 0;
    		}
        dJdW1.push_back(row);
        row.clear();
    	}
    }

    void test(std::vector<double> x, int r) {
      std::cout << "Resultado esperado: " << r << std::endl;
      double val = 0;
      // std::vector<std::vector<double>> transx;
      std::vector<double> row;

      z2.clear();
      for(int i=0; i<1; i++) {
    		for(int j=0; j<25; j++) {
    			for(int k=0; k<400; k++) {
            val += x[k] * dJdW1[k][j];
    				// z2[i][j] += x[i][k] * W1[k][j];
    			}
          row.push_back(val);
          val = 0;
    		}
        z2.push_back(row);
        row.clear();
    	}

      a2.clear();
      for(int i=0; i<1; i++) {
    		for(int j=0; j<25; j++) {
          row.push_back(g(z2[i][j]));
          // a2[i][j] = g(z2[i][j]);
        }
        a2.push_back(row);
        row.clear();
      }

      z3.clear();
      for(int i=0; i<1; i++) {
    		for(int j=0; j<10; j++) {
    			for(int k=0; k<25; k++) {
            val += a2[i][k] * dJdW2[k][j];
    				// z3[i][j] += a2[i][k] * W2[k][j];
    			}
          row.push_back(val);
          val = 0;
    		}
        z3.push_back(row);
        row.clear();
    	}

      yhat.clear();
      for(int i=0; i<1; i++) {
    		for(int j=0; j<10; j++) {
          row.push_back(g(z3[i][j]));
        }
        yhat.push_back(row);
        row.clear();
      }

      for(int i=0; i<10; i++) {
    		std::cout << i+1 << ": " << yhat[0][i] << std::endl;
      }
    }

    void randInicializacionPesos() {
      float LO = -0.12;
      float HI = 0.12;
      std::vector<double> row;

      srand (static_cast <unsigned> (time(0)));

      for(int i=0; i<input_layer_size; i++) {
        for (int j=0; j<hidden_layer_size; j++) {
          row.push_back(LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO))));
        }
        W1.push_back(row);
        row.clear();
      }

      for(int i=0; i<hidden_layer_size; i++) {
        for (int j=0; j<output_layer_size; j++) {
          row.push_back(LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO))));
        }
        W2.push_back(row);
        row.clear();
      }
    }
};

int main() {
  // float x[5000][400];
  NeuralNetwork nn;

  auto start_cpu =  std::chrono::high_resolution_clock::now();
  nn.randInicializacionPesos();

  int counter;
  int row = 0;
  std::ifstream file("digitos.txt");
	std::string line;
  std::vector<std::vector<double>> xv;
  std::vector<std::vector<double>> yv;
  std::vector<std::string> values;
	std::vector<double> labels;
	char split_char = ' ';
	while (std::getline(file, line))
	{
		counter = 0;

		std::istringstream split(line);
		for (std::string each; std::getline(split, each, split_char); counter <= 400 ? values.push_back(each) : labels.push_back(stoi(each)))
			counter++;

    std::vector<double> doubleVector(values.size());
    std::transform(values.begin(), values.end(), doubleVector.begin(), [](const std::string& val)
    {
      return std::stof(val);
    });
    std::vector<double> r {0,0,0,0,0,0,0,0,0,0};
    r[labels[0]] = 1;
    yv.push_back(labels);
    xv.push_back(doubleVector);
    values.clear();
    labels.clear();
	}
  auto rng = std::default_random_engine {};

  std::vector<std::vector<double>> xv1;
  std::vector<std::vector<double>> yv1;
  std::vector<std::vector<double>> xv2;
  std::vector<std::vector<double>> yv2;
  // std::random_shuffle(xv.begin(), xv.end());
  // std::random_shuffle(yv.begin(), yv.end());
  std::shuffle(xv.begin(), xv.end(), rng);
  std::shuffle(yv.begin(), yv.end(), rng);

  for (int i=0; i<5000; i++) {
    if (i >= 4000) {
      xv2.push_back(xv[i]);
      yv2.push_back(yv[i]);
    }
    else {
      xv1.push_back(xv[i]);
      yv1.push_back(yv[i]);
    }
  }

  // std::cout << "hola" << std::endl;
  nn.entrenaRN(xv1, yv1);

  for (int i=0; i<5; i++) {
    nn.entrenaRN2(xv1, yv1);
  }

  auto end_cpu =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

  printf("Neural Network Training elapsed %f ms\n", duration_ms.count());

  // std::ifstream file2("nueve.txt");
  // xv.clear();
  // yv.clear();
  // // std::vector<std::string> values;
	// // std::vector<int> labels;
	// // char split_char = ' ';
	// while (std::getline(file2, line))
	// {
	// 	counter = 0;
  //
	// 	std::istringstream split(line);
	// 	for (std::string each; std::getline(split, each, split_char); counter <= 400 ? values.push_back(each) : labels.push_back(stoi(each)))
	// 		counter++;
  //
  //   std::vector<double> doubleVector(values.size());
  //   std::transform(values.begin(), values.end(), doubleVector.begin(), [](const std::string& val)
  //   {
  //     return std::stof(val);
  //   });
  //   yv.push_back(labels);
  //   xv.push_back(doubleVector);
  //   values.clear();
  //   labels.clear();
	// }

  start_cpu =  std::chrono::high_resolution_clock::now();
  for (int i=0; i<1000; i++) {
    nn.test(xv2[i], yv2[i][0]);
  }

  end_cpu =  std::chrono::high_resolution_clock::now();
  duration_ms = end_cpu - start_cpu;

  printf("Neural Network Testing elapsed %f ms\n", duration_ms.count());

  return 0;
}
