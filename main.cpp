#include <iostream>
#include <vector>
#include <stdlib.h>
#include <sstream>
#include <stdio.h>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
#include <math.h>
#include <map>
#include <set>
#include <fstream>
#include <string>

using namespace std;

double delta_D = 0.05;

int sample_index = 0;

int train_index = 0;

float damp_weight = 0.1;

template < typename T >
T sigmoid1(T x)
{
    return 1.0f / (1.0f + exp(-x));
}

template < typename T >
T dsigmoid1(T x)
{
    return (1.0f - x)*x;
}

template < typename T >
T sigmoid3(T x)
{
    return atan(x);
}

template < typename T >
T dsigmoid3(T x)
{
    return 1.00/(1+x*x);
}

template < typename T >
T sigmoid2(T x)
{
    return log(1+exp(1.00*x));
}

template < typename T >
T dsigmoid2(T x)
{
    return 1.00/(1+exp(-1.00*x));
}

template < typename T >
T sigmoid(T x,int type)
{
    switch(type)
    {
        case 0:
            return sigmoid1(x);
        case 1:
            return sigmoid2(x);
        case 2:
            return sigmoid3(x);
    }
}

template < typename T >
T dsigmoid(T x,int type)
{
    switch(type)
    {
        case 0:
            return dsigmoid1(x);
        case 1:
            return dsigmoid2(x);
        case 2:
            return dsigmoid3(x);
    }
}

template < typename T >
T max(T a,T b)
{
    return (a>b)?a:b;
}

int maxi(int a,int b)
{
    return (a>b)?a:b;
}

template < typename T >
void apply_worker(std::vector<long> const & indices,long size,T * y,T * W,T * x)
{
  for(long k=0;k<indices.size();k++)
  {
    long i = indices[k];
    y[i] = 0;
    for(long j=0;j<size;j++)
    {
      y[i] += W[i*size+j]*x[j];
    }
  }
}

template < typename T >
void outer_product_worker(std::vector<long> const & indices,long size,T * H,T * A,T * B,T fact)
{
  for(long k=0;k<indices.size();k++)
  {
    long i = indices[k];
    for(long j=0;j<size;j++)
    {
      H[i*size+j] += A[i] * B[j] * fact;
    }
  }
}

template<typename T>
struct quasi_newton_info
{
    quasi_newton_info()
    {
        quasi_newton_update = false;
    }

    long get_size()
    {
        long size = 0;
        for(long layer = 0;layer < n_layers;layer++)
        {
            size += n_nodes[layer+1]*n_nodes[layer] + n_nodes[layer+1];
        }
        return size;
    }

    void init_gradient ()
    {
        long size = get_size();
        for(long layer = 0,k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    grad_tmp[k] = 0;
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                grad_tmp[k] = 0;
            }
        }
    }

    void copy (T * src,T * dst,long size)
    {
        for(long k=0;k<size;k++)
        {
            dst[k] = src[k];
        }
    }

    void copy_avg (T * src,T * dst,T alph,long size)
    {
        for(long k=0;k<size;k++)
        {
            dst[k] += (src[k]-dst[k])*alph;
        }
    }

    bool quasi_newton_update;
    long n_layers;
    T *** weights_neuron;
    T **  weights_bias;
    std::vector<long> n_nodes;
    T * grad_tmp;
    T * grad_1;
    T * grad_2;
    T * Y;
    T * dX;
    T * B;
    T * H;
    T alpha;

    void init_QuasiNewton()
    {
        long size = get_size();
        grad_tmp = new T[size];
        init_gradient();
        grad_1 = new T[size];
        grad_2 = new T[size];
        copy(grad_tmp,grad_1,size);
        copy(grad_tmp,grad_2,size);
        B = new T[size*size];
        T * B_tmp = init_B();
        copy(B_tmp,B,size*size);
        delete [] B_tmp;
        H = new T[size*size];
        T * H_tmp = init_H();
        copy(H_tmp,H,size*size);
        delete [] H_tmp;
        dX = new T[size*size];
        T * dX_tmp = get_dx();
        copy(dX_tmp,dX,size);
        delete [] dX_tmp;
        Y = new T[size*size];
    }

    T * init_B()
    {
        long size = get_size();
        T * B = new T[size*size];
        for(long t=0;t<size*size;t++)
        {
            B[t] = 0;
        }
        for(long layer = 0, k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    B[k*size+k] = weights_neuron[layer][i][j];
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                B[k*size+k] = weights_bias[layer][i];
            }
        }
        return B;
    }

    T * init_H()
    {
        long size = get_size();
        T * H = new T[size*size];
        for(long t=0;t<size*size;t++)
        {
            H[t] = 0;
        }
        for(long layer = 0, k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    H[k*size+k] = -1;
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                H[k*size+k] = -1;
            }
        }
        return H;
    }

    void update_QuasiNewton()
    {
        long size = get_size();
        copy_avg(grad_2,grad_1,0.1,size);
        copy(grad_tmp,grad_2,size);
        T * Y_tmp = get_y();
        copy(Y_tmp,Y,size);
        delete [] Y_tmp;
        T * dX_tmp = get_dx();
        copy(dX_tmp,dX,size);
        delete [] dX_tmp;
    }

    T * get_y ()
    {
        long size = get_size();
        T * y = new T[size];
        //T y_m = 0;
        for(long k=0;k<size;k++)
        {
            y[k] = grad_2[k] - grad_1[k];
            //y_m = max(y_m,fabs(y[k]));
        }
        return y;
    }

    T * get_dx ()
    {
        long size = get_size();
        T * dx = apply(H,grad_1);
        for(long k=0;k<size;k++)
        {
            dx[k] *= -alpha;
        }
        return dx;
    }

    T * get_outer_product(T * a,T * b)
    {
        long size = get_size();
        long prod_size = size*size;
        T * prod = new T[prod_size];
        for(long i=0,k=0;i<size;i++)
        {
          for(long j=0;j<size;j++,k++)
          {
            prod[k] = a[i]*b[j];
          }
        }
        return prod;
    }

    T get_inner_product(T * a,T * b)
    {
        T ret = 0;
        long size = get_size();
        for(long i=0;i<size;i++)
        {
            ret += a[i]*b[i];
        }
        T eps = 1e-2;
        if(ret<0)
        {
            ret -= eps;
        }
        else
        {
            ret += eps;
        }
        return ret;
    }

    T * apply(T * W, T * x)
    {
        long size = get_size();
        T * y = new T[size];
        std::vector<boost::thread * > threads;
        long num_cpu = boost::thread::hardware_concurrency();
        std::vector<std::vector<long> > indices(num_cpu);
        for(long i=0;i<size;i++)
        {
          indices[i%num_cpu].push_back(i);
        }
        for(long i=0;i<num_cpu;i++)
        {
          threads.push_back(new boost::thread(apply_worker<T>,indices[i],size,&y[0],&W[0],&x[0]));
        }
        for(long i=0;i<threads.size();i++)
        {
          threads[i]->join();
          delete threads[i];
        }
        return y;
    }

    T * apply_t(T * x, T * W)
    {
        long size = get_size();
        T * y = new T[size];
        for(long i=0,k=0;i<size;i++)
        {
          y[i] = 0;
          for(long j=0;j<size;j++,k++)
          {
            y[i] += W[size*j+i]*x[j];
          }
        }
        return y;
    }

    T limit(T x,T eps)
    {
        if(x>0)
        {
            if(x>eps)return eps;
        }
        else
        {
            if(x<-eps)return -eps;
        }
        return x;
    }

    // SR1
    void SR1_update()
    {
        long size = get_size();
        T * dx_Hy = apply(H,Y);
        for(long i=0;i<size;i++)
        {
          dx_Hy[i] = dX[i] - dx_Hy[i];
        }
        T inner = 1.0 / (get_inner_product(dx_Hy,Y));
        std::vector<boost::thread * > threads;
        long num_cpu = boost::thread::hardware_concurrency();
        std::vector<std::vector<long> > indices(num_cpu);
        for(long i=0;i<size;i++)
        {
          indices[i%num_cpu].push_back(i);
        }
        for(long i=0;i<num_cpu;i++)
        {
          threads.push_back(new boost::thread(outer_product_worker<T>,indices[i],size,&H[0],&dx_Hy[0],&dx_Hy[0],inner));
        }
        for(long i=0;i<threads.size();i++)
        {
          threads[i]->join();
          delete threads[i];
        }
        delete [] dx_Hy;
    }

    // Broyden
    void Broyden_update()
    {
        long size = get_size();
        T * dx_Hy = apply(H,Y);
        for(long i=0;i<size;i++)
        {
          dx_Hy[i] = dX[i] - dx_Hy[i];
        }
        T * xH = apply_t(dX,H);
        T * outer = get_outer_product(dx_Hy,xH);
        T inner = 1.0 / (get_inner_product(xH,Y));
        for(long i=0;i<size*size;i++)
        {
          H[i] += outer[i] * inner;
        }
        delete [] dx_Hy;
        delete [] xH;
        delete [] outer;
    }

    // DFP
    void DFP_update()
    {
        long size = get_size();
        T * Hy = apply(H,Y);
        T * outer_2 = get_outer_product(Hy,Hy);
        T inner_2 = -1.0 / (get_inner_product(Hy,Y));
        T * outer_1 = get_outer_product(dX,dX);
        T inner_1 = 1.0 / (get_inner_product(dX,Y));
        for(long i=0;i<size*size;i++)
        {
          H[i] += outer_1[i] * inner_1 + outer_2[i] * inner_2;
        }
        delete [] outer_2;
        delete [] outer_1;
        delete [] Hy;
    }

    T * apply_M(T * A, T * B)
    {
        long size = get_size();
        T * C = new T[size*size];
        for(long i=0,k=0;i<size;i++)
        {
          for(long j=0;j<size;j++,k++)
          {
            C[k] = 0;
            for(long t=0;t<size;t++)
            {
              C[k] += A[i*size+t]*B[t*size+j];
            }
          }
        }
        return C;
    }

    // BFGS
    void BFGS_update()
    {
        long size = get_size();
        T inner = 1.0 / (get_inner_product(Y,dX));
        T * outer_xx = get_outer_product(dX,dX);
        T * outer_xy = get_outer_product(dX,Y);
        T * outer_yx = get_outer_product(Y,dX);
        for(long i=0,k=0;i<size;i++)
        {
          for(long j=0;j<size;j++,k++)
          {
            if(i==j)
            {
              outer_xy[k] = 1-outer_xy[k]*inner;
              outer_yx[k] = 1-outer_yx[k]*inner;
            }
            else
            {
              outer_xy[k] = -outer_xy[k]*inner;
              outer_yx[k] = -outer_yx[k]*inner;
            }
            outer_xx[k] = outer_xx[k]*inner;
          }
        }
        T * F = apply_M(outer_xy,H);
        T * G = apply_M(F,outer_yx);
        for(long i=0;i<size*size;i++)
        {
          H[i] = G[i] + outer_xx[i];
        }
        delete [] F;
        delete [] G;
        delete [] outer_xx;
        delete [] outer_xy;
        delete [] outer_yx;
    }

};

template<typename T>
struct training_info
{

    quasi_newton_info<T> * quasi_newton;

    std::vector<long> n_nodes;
    T **  activation_values;
    T **  deltas;
    long n_variables;
    long n_labels;
    long n_layers;
    long n_elements;

    T *** weights_neuron;
    T **  weights_bias;
    T *** partial_weights_neuron;
    T **  partial_weights_bias;

    T partial_error;
    T smallest_index;

    T epsilon;

    int type;

    training_info()
    {

    }

    void init(T _alpha)
    {
        type = 0;
        smallest_index = 0;
        partial_error = 0;
        activation_values  = new T*[n_nodes.size()];
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            activation_values [layer] = new T[n_nodes[layer]];
        }
        deltas = new T*[n_nodes.size()];
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            deltas[layer] = new T[n_nodes[layer]];
        }
        partial_weights_neuron = new T**[n_layers];
        partial_weights_bias = new T*[n_layers];
        for(long layer = 0;layer < n_layers;layer++)
        {
            partial_weights_neuron[layer] = new T*[n_nodes[layer+1]];
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                partial_weights_neuron[layer][i] = new T[n_nodes[layer]];
                for(long j=0;j<n_nodes[layer];j++)
                {
                    partial_weights_neuron[layer][i][j] = 0;
                }
            }
            partial_weights_bias[layer] = new T[n_nodes[layer+1]];
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                partial_weights_bias[layer][i] = 0;
            }
        }
    }

    void destroy()
    {
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            delete [] activation_values [layer];
        }
        delete [] activation_values;
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            delete [] deltas [layer];
        }
        delete [] deltas;
        for(long layer = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                delete [] partial_weights_neuron[layer][i];
            }
            delete [] partial_weights_neuron[layer];
        }
        delete [] partial_weights_neuron;
        for(long layer = 0;layer < n_layers;layer++)
        {
            delete [] partial_weights_bias[layer];
        }
        delete [] partial_weights_bias;
    }

    void update_gradient ()
    {
        for(long layer = 0,k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    quasi_newton->grad_tmp[k] += partial_weights_neuron[layer][i][j] / n_elements;
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                quasi_newton->grad_tmp[k] += partial_weights_bias[layer][i] / n_elements;
            }
        }
    }

    void globalUpdate()
    {
        if(quasi_newton->quasi_newton_update)
        {
            for(long layer = 0,k = 0;layer < n_layers;layer++)
            {
                for(long i=0;i<n_nodes[layer+1];i++)
                {
                    for(long j=0;j<n_nodes[layer];j++,k++)
                    {
                        weights_neuron[layer][i][j] += quasi_newton->dX[k];
                    }
                }
                for(long i=0;i<n_nodes[layer+1];i++,k++)
                {
                    weights_bias[layer][i] += quasi_newton->dX[k];
                }
            }
        }
        else
        {
            for(long layer = 0,k = 0;layer < n_layers;layer++)
            {
                for(long i=0;i<n_nodes[layer+1];i++)
                {
                    for(long j=0;j<n_nodes[layer];j++,k++)
                    {
                        weights_neuron[layer][i][j] += epsilon * quasi_newton->grad_tmp[k];
                    }
                }
                for(long i=0;i<n_nodes[layer+1];i++,k++)
                {
                    weights_bias[layer][i] += epsilon * quasi_newton->grad_tmp[k];
                }
            }
        }
    }

};

template<typename T>
T min(T a,T b)
{
    return (a<b)?a:b;
}

template<typename T>
void training_worker(training_info<T> * g,std::vector<long> const & vrtx,T * variables,T * labels)
{
    
    for(long n=0;n<vrtx.size();n++)
    {

        // initialize input activations
        for(long i=0;i<g->n_nodes[0];i++)
        {
            g->activation_values[0][i] = variables[vrtx[n]*g->n_variables+i];
            // HUUUGGGGEEEE  MARK !!!!!!!!!!!!!!!!!!
            if(i==0)g->activation_values[0][i] = 0.5;
        }
        // forward propagation
        for(long layer = 0; layer < g->n_layers; layer++)
        {
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                T sum = g->weights_bias[layer][i];
                for(long j=0;j<g->n_nodes[layer];j++)
                {
                    sum += g->activation_values[layer][j] * g->weights_neuron[layer][i][j];
                }
                g->activation_values[layer+1][i] = sigmoid(sum,g->type);
                //std::cout << g->activation_values[layer+1][i] << '\t';
            }
            //std::cout << std::endl;
        }
        long last_layer = g->n_nodes.size()-2;
        // initialize observed labels
        T max_err = 0;
        T min_err = 1e12;
        T tmp_err;
        T min_partial_error = 0;
        T ind = 0;
        for(long i=0;i<g->n_nodes[last_layer];i++)
        {
            tmp_err = fabs(g->deltas[last_layer+1][i] = labels[vrtx[n]*g->n_labels+i] - g->activation_values[last_layer][i]);
            if(tmp_err>max_err)
            {
                max_err = tmp_err;
            }
            if(tmp_err<min_err)
            {
                min_err = tmp_err;
                ind = i;
            }
        }
        // MARK !!!!!!!!!!!!!!!!!!
        train_index = 0;
        for(long i=0;i<g->n_nodes[last_layer];i++)
        {
            //g->partial_error += fabs(g->deltas[last_layer+1][i]);
            //if(i==sample_index)
            //if(fabs(g->deltas[last_layer+1][i]<min_partial_error))
            //{
            //    min_partial_error = fabs(g->deltas[last_layer+1][i]);
            //    ind = i;
            //}
            g->deltas[last_layer+1][i] = 0;
            if(i==train_index)
            {
                g->deltas[last_layer+1][i] = labels[vrtx[n]*g->n_labels+i] - g->activation_values[last_layer][i];
                min_partial_error += fabs(g->deltas[last_layer+1][i]);
            }
            else
            {
                g->deltas[last_layer+1][i] = 0;//damp_weight*(labels[vrtx[n]*g->n_labels+i] - g->activation_values[last_layer][i]);
                //min_partial_error += fabs(g->deltas[last_layer+1][i]);
            }
            //std::cout << g->deltas[last_layer+1][i] << '\t';
        }
        g->partial_error += min_partial_error;
        g->smallest_index += ind;
        //std::cout << std::endl;
        // back propagation
        for(long layer = g->n_layers-1; layer >= 0; layer--)
        {
            // back propagate deltas
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                g->deltas[layer+1][i] = 0;
                for(long j=0;j<g->n_nodes[layer+2];j++)
                {
                    if(layer+1==last_layer)
                    {
                        g->deltas[layer+1][i] += dsigmoid(g->activation_values[layer+1][i],g->type)*g->deltas[layer+2][j];
                    }
                    else
                    {
                        g->deltas[layer+1][i] += dsigmoid(g->activation_values[layer+1][i],g->type)*g->deltas[layer+2][j]*g->weights_neuron[layer+1][j][i];
                    }
                }
                //std::cout << g->deltas[layer+1][i] << '\t';
            }
            //std::cout << std::endl;
            //std::cout << "biases" << std::endl;
            // biases
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                g->partial_weights_bias[layer][i] += g->deltas[layer+1][i];
                //std::cout << g->partial_weights_bias[layer][i] << '\t';
            }
            //std::cout << std::endl;
            //std::cout << "neuron weights" << std::endl;
            // neuron weights
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                for(long j=0;j<g->n_nodes[layer];j++)
                {
                    g->partial_weights_neuron[layer][i][j] += g->activation_values[layer][j] * g->deltas[layer+1][i];
                    //std::cout << g->partial_weights_neuron[layer][i][j] << '\t';
                }
                //std::cout << std::endl;
            }
            //std::cout << std::endl;
        }
        //char ch;
        //std::cin >> ch;
    }

}

bool stop_training = false;
bool continue_training = true;

std::vector<double> errs;
std::vector<double> test_errs;

template<typename T>
struct Perceptron
{
    quasi_newton_info<T> * quasi_newton;

    T ierror;
    T perror;

    T *** weights_neuron;
    T **  weights_bias;
    T **  activation_values;
    T **  activation_values1;
    T **  activation_values2;
    T **  activation_values3;
    T **  deltas;

    long n_inputs;
    long n_outputs;
    long n_layers;
    std::vector<long> n_nodes;

    T get_variable(int ind)
    {
        int I = 0;
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
              if(I==ind)return weights_bias[layer][i];
              I++;
            }
          }
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(int i=0;i<n_nodes[layer+1];i++)
            {
                for(int j=0;j<n_nodes[layer];j++)
                {
                    if(I==ind)return weights_neuron[layer][i][j];
                    I++;
                }
            }
          }
        return 0;
    }

    int get_num_variables()
    {
        int I = 0;
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
              I++;
            }
          }
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(int i=0;i<n_nodes[layer+1];i++)
            {
                for(int j=0;j<n_nodes[layer];j++)
                {
                    I++;
                }
            }
          }
        return I;
    }

    void dump_to_file(std::string filename,bool quiet=false)
    {
        if(!quiet)
          std::cout << "dump to file:" << filename << std::endl;
        ofstream myfile (filename.c_str());
        if (myfile.is_open())
        {
          myfile << "#n_nodes" << std::endl;
          myfile << n_nodes.size() << " ";
          for(int i=0;i<n_nodes.size();i++)
          {
            myfile << n_nodes[i] << " ";
          }
          myfile << std::endl;
          myfile << "#bias" << std::endl;
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
              myfile << (float)weights_bias[layer][i] << " ";
            }
            std::cout << " ";
          }
          myfile << std::endl;
          myfile << "#weights" << std::endl;
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(int i=0;i<n_nodes[layer+1];i++)
            {
                for(int j=0;j<n_nodes[layer];j++)
                {
                    myfile << (float)weights_neuron[layer][i][j] << " ";
                }
            }
          }
          myfile << std::endl;
          myfile << "#error" << std::endl;
          myfile << final_error << std::endl;
          myfile.close();
        }
        else
        {
          cout << "Unable to open file: " << filename << std::endl;
          exit(1);
        }

    }

    void load_from_file(std::string filename,bool quiet=false)
    {
        if(!quiet)
          std::cout << "loading from file:" << filename << std::endl;
        ifstream myfile (filename.c_str());
        if (myfile.is_open())
        {
          std::string line;
          std::string tmp;
          int stage = 0;
          bool done = false;
          while(!done&&getline(myfile,line))
          {
            if(line[0] == '#')continue;
            switch(stage)
            {
              case 0: // get n_nodes
              {
                std::stringstream ss;
                ss << line;
                int n_nodes_size;
                ss >> tmp;
                n_nodes_size = atoi(tmp.c_str());
                if(n_nodes_size != n_nodes.size())
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                for(int i=0;i<n_nodes_size;i++)
                {
                  int layer_size;
                  ss >> tmp;
                  layer_size = atoi(tmp.c_str());
                  if(layer_size != n_nodes[i])
                  {
                    std::cout << "network structure is not consistent." << std::endl;
                    exit(1);
                  }
                }
                stage = 1;
                break;
              }
              case 1: // get bias
              {
                std::stringstream ss;
                ss << line;
                for(int layer = 0;layer < n_layers;layer++)
                {
                  for(long i=0;i<n_nodes[layer+1];i++)
                  {
                    ss >> tmp;
                    weights_bias[layer][i] = atof(tmp.c_str());
                  }
                }
                stage = 2;
                break;
              }
              case 2: // get weights
              {
                std::stringstream ss;
                ss << line;
                for(int layer = 0;layer < n_layers;layer++)
                {
                  for(int i=0;i<n_nodes[layer+1];i++)
                  {
                      for(int j=0;j<n_nodes[layer];j++)
                      {
                          ss >> tmp;
                          weights_neuron[layer][i][j] = atof(tmp.c_str());
                      }
                  }
                }
                stage = 3;
                break;
              }
              case 3: // final error
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                final_error = atof(tmp.c_str());
                stage = 4;
                break;
              }
              default:done = true;break;
            }
          }
          myfile.close();
        }
        else cout << "Unable to open file: " << filename << std::endl;

    }

    T epsilon;
    T alpha;
    int sigmoid_type;

    // std::vector<long> nodes;
    // nodes.push_back(2); // inputs
    // nodes.push_back(3); // hidden layer
    // nodes.push_back(1); // output layer
    // nodes.push_back(1); // outputs
    Perceptron(std::vector<long> p_nodes)
    {

        quasi_newton = NULL;

        sigmoid_type = 0;
        alpha = 0.1;

        ierror = 1e10;
        perror = 1e10;

        n_nodes = p_nodes;
        n_inputs = n_nodes[0];
        n_outputs = n_nodes[n_nodes.size()-1];
        n_layers = n_nodes.size()-2; // first and last numbers and output and input dimensions, so we have n-2 layers

        weights_neuron = new T**[n_layers];
        weights_bias = new T*[n_layers];
        activation_values  = new T*[n_nodes.size()];
        activation_values1 = new T*[n_nodes.size()];
        activation_values2 = new T*[n_nodes.size()];
        activation_values3 = new T*[n_nodes.size()];
        deltas = new T*[n_nodes.size()];
        
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            activation_values [layer] = new T[n_nodes[layer]];
            activation_values1[layer] = new T[n_nodes[layer]];
            activation_values2[layer] = new T[n_nodes[layer]];
            activation_values3[layer] = new T[n_nodes[layer]];
            deltas[layer] = new T[n_nodes[layer]];
        }

        for(long layer = 0;layer < n_layers;layer++)
        {
            weights_neuron[layer] = new T*[n_nodes[layer+1]];
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                weights_neuron[layer][i] = new T[n_nodes[layer]];
                for(long j=0;j<n_nodes[layer];j++)
                {
                    weights_neuron[layer][i][j] = 1.0 * (-1.0 + 2.0 * ((rand()%10000)/10000.0));
                }
            }
            weights_bias[layer] = new T[n_nodes[layer+1]];
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                weights_bias[layer][i] = 1.0 * (-1.0 + 2.0 * ((rand()%10000)/10000.0));
            }
        }

        //weights_neuron[0][0][0] = .1;        weights_neuron[0][0][1] = .2;
        //weights_neuron[0][1][0] = .3;        weights_neuron[0][1][1] = .4;
        //weights_neuron[0][2][0] = .5;        weights_neuron[0][2][1] = .6;

        //weights_bias[0][0] = .1;
        //weights_bias[0][1] = .2;
        //weights_bias[0][2] = .3;

        //weights_neuron[1][0][0] = .6;        weights_neuron[1][0][1] = .7;      weights_neuron[1][0][2] = .8;

        //weights_bias[1][0] = .5;

    }

    T * model(long n_elements,long n_labels,T * variables)
    {
        T * labels = new T[n_labels];
        // initialize input activations
        for(long i=0;i<n_nodes[0];i++)
        {
            activation_values1[0][i] = variables[i];
        }
        // forward propagation
        for(long layer = 0; layer < n_layers; layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                T sum = weights_bias[layer][i];
                for(long j=0;j<n_nodes[layer];j++)
                {
                    sum += activation_values1[layer][j] * weights_neuron[layer][i][j];
                }
                activation_values1[layer+1][i] = sigmoid(sum,0);// <- zero is important here!!!!
            }
        }
        long last_layer = n_nodes.size()-2;
        for(long i=0;i<n_labels;i++)
        {
            labels[i] = activation_values1[last_layer][i];
        }
        return labels;
    }

    T * model2(long n_elements,long n_labels,T * variables)
    {
        T * labels = new T[n_labels];
        // initialize input activations
        for(long i=0;i<n_nodes[0];i++)
        {
            activation_values3[0][i] = variables[i];
        }
        // forward propagation
        for(long layer = 0; layer < n_layers; layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                T sum = weights_bias[layer][i];
                for(long j=0;j<n_nodes[layer];j++)
                {
                    sum += activation_values3[layer][j] * weights_neuron[layer][i][j];
                }
                activation_values3[layer+1][i] = sigmoid(sum,0);// <- zero is important here!!!!
            }
        }
        long last_layer = n_nodes.size()-2;
        for(long i=0;i<n_labels;i++)
        {
            labels[i] = activation_values3[last_layer][i];
        }
        return labels;
    }

    T verify( long n_test_elements
            , long n_variables
            , T * test_variables
            , long n_labels
            , T * test_labels
            )
    {
        T err = 0;

        T * labels = new T[n_labels];

        for(int e=0;e<n_test_elements;e++)
        {

          // initialize input activations
          for(long i=0;i<n_variables;i++)
          {
              activation_values2[0][i] = test_variables[e*n_variables+i];
          }
          // forward propagation
          for(long layer = 0; layer < n_layers; layer++)
          {
              for(long i=0;i<n_nodes[layer+1];i++)
              {
                  T sum = weights_bias[layer][i];
                  for(long j=0;j<n_nodes[layer];j++)
                  {
                      sum += activation_values2[layer][j] * weights_neuron[layer][i][j];
                  }
                  activation_values2[layer+1][i] = sigmoid(sum,0);// <- zero is important here!!!!
              }
          }
          long last_layer = n_nodes.size()-2;
          for(long i=0;i<n_labels;i++)
          {
            if(i==sample_index)
            {
              err += fabs(test_labels[e*n_labels+i] - activation_values2[last_layer][i]);
            }
          }

        }

        delete [] labels;

        return err/n_test_elements;

    }

    int get_sigmoid()
    {
        return sigmoid_type;
    }

    void train ( int p_sigmoid_type
               , T p_epsilon
               , long n_iterations
               , long n_elements
               , long n_test_elements
               , long n_variables
               , T * variables
               , T * test_variables
               , long n_labels
               , T * labels
               , T * test_labels
               , quasi_newton_info<T> * q_newton = NULL
               )
    {
        sigmoid_type = p_sigmoid_type;
        epsilon = p_epsilon;
        if(n_variables != n_nodes[0]){std::cout << "error 789437248932748293" << std::endl;exit(0);}
        if(q_newton == NULL)
        {
            quasi_newton = new quasi_newton_info<T>();
            quasi_newton->alpha = alpha;
            quasi_newton->n_nodes = n_nodes;
            quasi_newton->n_layers = n_layers;
            quasi_newton->weights_neuron = weights_neuron;
            quasi_newton->weights_bias = weights_bias;
            quasi_newton->init_QuasiNewton();
            quasi_newton->quasi_newton_update = false;
        }
        else
        {
            quasi_newton = q_newton;
        }
        ierror = 1e10;
        bool init = true;
        perror = 1e10;
        T min_final_error = 1e10;
        for(long iter = 0; iter < n_iterations || continue_training; iter++)
        {
            T error = 0;
            T index = 0;

            //////////////////////////////////////////////////////////////////////////////////
            //                                                                              //
            //          Multi-threaded block                                                //
            //                                                                              //
            //////////////////////////////////////////////////////////////////////////////////
            std::vector<boost::thread*> threads;
            std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
            std::vector<training_info<T>*> g;
            for(long i=0;i<n_elements;i++)
            {
              vrtx[i%vrtx.size()].push_back(i);
            }
            for(long i=0;i<vrtx.size();i++)
            {
              g.push_back(new training_info<T>());
            }
            quasi_newton->init_gradient();
            for(long thread=0;thread<vrtx.size();thread++)
            {
              g[thread]->quasi_newton = quasi_newton;
              g[thread]->n_nodes = n_nodes;
              g[thread]->n_elements = n_elements;
              g[thread]->n_variables = n_variables;
              g[thread]->n_labels = n_labels;
              g[thread]->n_layers = n_layers;
              g[thread]->weights_neuron = weights_neuron;
              g[thread]->weights_bias = weights_bias;
              g[thread]->epsilon = epsilon;
              g[thread]->type = get_sigmoid();

              g[thread]->init(alpha);
              threads.push_back(new boost::thread(training_worker<T>,g[thread],vrtx[thread],variables,labels));
            }
            for(long thread=0;thread<vrtx.size();thread++)
            {
              threads[thread]->join();
              g[thread]->update_gradient();
              delete threads[thread];
            }
            quasi_newton->update_QuasiNewton();
            quasi_newton->SR1_update();
            for(long thread=0;thread<vrtx.size();thread++)
            {
              g[thread]->globalUpdate();
              error += g[thread]->partial_error;
              index += g[thread]->smallest_index;
              g[thread]->destroy();
              delete g[thread];
            }
            threads.clear();
            vrtx.clear();
            g.clear();
            final_error = verify(n_test_elements,n_variables,test_variables,n_labels,test_labels);
            static int cnt1 = 0;
            if(cnt1%100==0)
            std::cout << iter << "\ttrain index=" << train_index << "\tdamp weight=" << damp_weight << "\tquasi_newton_update=" << quasi_newton->quasi_newton_update << "\ttype=" << sigmoid_type << "\tepsilon=" << epsilon << "\talpha=" << alpha << '\t' << "error=" << error << "\tdiff=" << (error-perror) << "\t\%error=" << 100*error/n_elements << "\ttest\%error=" << 100*final_error << "\tindex=" << index/n_elements << std::endl;
            cnt1++;
            perror = error;
            errs.push_back(error/n_elements);
            test_errs.push_back(final_error);
            if(error/n_elements < 0.1)
            {
                train_index = (train_index+1)%n_labels;//round(index/n_elements);
            }
            //if(train_index<0||train_index>=n_elements-1)
            //{
            //    train_index = 0;
            //}
            if(init)
            {
                ierror = error;
                init = false;
            }

            if((iter+1)%100000==0||stop_training)
            {
                std::stringstream ss;
                ss << "snapshots/network.ann." << ((int)(100*10000*final_error)/10000.0f);
                dump_to_file(ss.str());
            }

            if(stop_training){stop_training=false;break;}

            // MARK!!!
            //if(error/n_elements < 0.05 && iter > n_iterations)
            //{
            //  std::stringstream ss;
            //  ss << "snapshots/network.ann." << ((int)(100*10000*final_error)/10000.0f);
            //  dump_to_file(ss.str());
            //  dump_to_file("network.ann");
            //  exit(1);
            //}

            //char ch;
            //std::cin >> ch;

        }
    }

    T final_error;

};


void clear() {
  // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
  std::cout << "\x1B[2J\x1B[H";
}

double norm(double * dat,long size)
{
  double ret = 0;
  for(long i=0;i<size;i++)
  {
    ret += dat[i]*dat[i];
  }
  return sqrt(ret);
}

void zero(double * dat,long size)
{
  for(long i=0;i<size;i++)
  {
    dat[i] = 0;
  }
}

void constant(double * dat,double val,long size)
{
  for(long i=0;i<size;i++)
  {
    dat[i] = (-1+2*((rand()%10000)/10000.0f))*val;
  }
}

void add(double * A, double * dA, double epsilon, long size)
{
  for(long i=0;i<size;i++)
  {
    A[i] += epsilon * dA[i];
  }
}

struct gradient_info
{
  long n;
  long v;
  long h;
  double * vis0;
  double * hid0;
  double * vis;
  double * hid;
  double * dW;
  double * dc;
  double * db;
  double partial_err;
  double * partial_dW;
  double * partial_dc;
  double * partial_db;
  void init()
  {
    partial_err = 0;
    partial_dW = new double[h*v];
    for(int i=0;i<h*v;i++)partial_dW[i]=0;
    partial_dc = new double[h];
    for(int i=0;i<h;i++)partial_dc[i]=0;
    partial_db = new double[v];
    for(int i=0;i<v;i++)partial_db[i]=0;
  }
  void destroy()
  {
    delete [] partial_dW;
    delete [] partial_dc;
    delete [] partial_db;
  }
  void globalUpdate()
  {
    for(int i=0;i<h*v;i++)
        dW[i] += partial_dW[i];
    for(int i=0;i<h;i++)
        dc[i] += partial_dc[i];
    for(int i=0;i<v;i++)
        db[i] += partial_db[i];
  }
};

void gradient_worker(gradient_info * g,std::vector<long> const & vrtx)
{
  double factor = 1.0f / g->n;
  double factorv= 1.0f / (g->v*g->v);
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long i=0;i<g->v;i++)
    {
      for(long j=0;j<g->h;j++)
      {
        g->partial_dW[i*g->h+j] -= factor * (g->vis0[k*g->v+i]*g->hid0[k*g->h+j] - g->vis[k*g->v+i]*g->hid[k*g->h+j]);
      }
    }

    for(long j=0;j<g->h;j++)
    {
      g->partial_dc[j] -= factor * (g->hid0[k*g->h+j]*g->hid0[k*g->h+j] - g->hid[k*g->h+j]*g->hid[k*g->h+j]);
    }

    for(long i=0;i<g->v;i++)
    {
      g->partial_db[i] -= factor * (g->vis0[k*g->v+i]*g->vis0[k*g->v+i] - g->vis[k*g->v+i]*g->vis[k*g->v+i]);
    }

    for(long i=0;i<g->v;i++)
    {
      g->partial_err += factorv * (g->vis0[k*g->v+i]-g->vis[k*g->v+i])*(g->vis0[k*g->v+i]-g->vis[k*g->v+i]);
    }
  }
}

void vis2hid_worker(const double * X,double * H,long h,long v,double * c,double * W,std::vector<long> const & vrtx)
{
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long j=0;j<h;j++)
    {
      H[k*h+j] = c[j]; 
      for(long i=0;i<v;i++)
      {
        H[k*h+j] += W[i*h+j] * X[k*v+i];
      }
      H[k*h+j] = 1.0f/(1.0f + exp(-H[k*h+j]));
    }
  }
}

void hid2vis_worker(const double * H,double * V,long h,long v,double * b,double * W,std::vector<long> const & vrtx)
{
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long i=0;i<v;i++)
    {
      V[k*v+i] = b[i]; 
      for(long j=0;j<h;j++)
      {
        V[k*v+i] += W[i*h+j] * H[k*h+j];
      }
      V[k*v+i] = 1.0f/(1.0f + exp(-V[k*v+i]));
    }
  }
}

struct RBM
{
  long h; // number hidden elements
  long v; // number visible elements
  long n; // number of samples
  double * c; // bias term for hidden state, R^h
  double * b; // bias term for visible state, R^v
  double * W; // weight matrix R^h*v
  double * X; // input data, binary [0,1], v*n

  double * vis0;
  double * hid0;
  double * vis;
  double * hid;
  double * dW;
  double * dc;
  double * db;

  RBM(long _v,long _h,double * _W,double * _b,double * _c,long _n,double * _X)
  {
    //for(long k=0;k<100;k++)
    //  std::cout << _X[k] << "\t";
    //std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = _c;
    b = _b;
    W = _W;

    vis0 = NULL;
    hid0 = NULL;
    vis = NULL;
    hid = NULL;
    dW = NULL;
    dc = NULL;
    db = NULL;
  }
  RBM(long _v,long _h,long _n,double* _X)
  {
    //for(long k=0;k<100;k++)
    //  std::cout << _X[k] << "\t";
    //std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = new double[h];
    b = new double[v];
    W = new double[h*v];
    constant(c,0.5f,h);
    constant(b,0.5f,v);
    constant(W,0.5f,v*h);

    vis0 = NULL;
    hid0 = NULL;
    vis = NULL;
    hid = NULL;
    dW = NULL;
    dc = NULL;
    db = NULL;
  }

  void init(int offset)
  {
    boost::posix_time::ptime time_start(boost::posix_time::microsec_clock::local_time());
    if(vis0==NULL)vis0 = new double[n*v];
    if(hid0==NULL)hid0 = new double[n*h];
    if(vis==NULL)vis = new double[n*v];
    if(hid==NULL)hid = new double[n*h];
    if(dW==NULL)dW = new double[h*v];
    if(dc==NULL)dc = new double[h];
    if(db==NULL)db = new double[v];

    //std::cout << "n*v=" << n*v << std::endl;
    //std::cout << "offset=" << offset << std::endl;
    for(long i=0,size=n*v;i<size;i++)
    {
      vis0[i] = X[i+offset];
    }

    vis2hid(vis0,hid0);
    boost::posix_time::ptime time_end(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration(time_end - time_start);
    //std::cout << "init timing:" << duration << '\n';
  }

  void cd(long nGS,double epsilon,int offset=0,bool bottleneck=false)
  {
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());
    //std::cout << "cd" << std::endl;

    // CD Contrastive divergence (Hlongon's CD(k))
    //   [dW, db, dc, act] = cd(self, X) returns the gradients of
    //   the weihgts, visible and hidden biases using Hlongon's
    //   approximated CD. The sum of the average hidden units
    //   activity is returned in act as well.

    for(long i=0;i<n*h;i++)
    {
      hid[i] = hid0[i];
    }
    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration10(time_1 - time_0);
    //std::cout << "cd timing 1:" << duration10 << '\n';

    for (long iter = 1;iter<=nGS;iter++)
    {
      //std::cout << "iter=" << iter << std::endl;
      // sampling
      hid2vis(hid,vis);
      vis2hid(vis,hid);

// Preview stuff
#if 0
      long off = dat_offset%(n);
      long offv = off*v;
      long offh = off*h;
      long off_preview = off*(3*WIN*WIN+10);
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis_preview[k] = vis[offv+k];
          vis_previewG[k] = vis[offv+k+WIN*WIN];
          vis_previewB[k] = vis[offv+k+2*WIN*WIN];
        }
      }
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis1_preview[k] = orig_arr[offset+off_preview+k];
          vis1_previewG[k] = orig_arr[offset+off_preview+k+WIN*WIN];
          vis1_previewB[k] = orig_arr[offset+off_preview+k+2*WIN*WIN];
        }
      }
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis0_preview[k] = vis0[offv+k];
          vis0_previewG[k] = vis0[offv+k+WIN*WIN];
          vis0_previewB[k] = vis0[offv+k+2*WIN*WIN];
        }
      }
#endif

    }
    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration21(time_2 - time_1);
    //std::cout << "cd timing 2:" << duration21 << '\n';
  
    zero(dW,v*h);
    zero(dc,h);
    zero(db,v);
    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration32(time_3 - time_2);
    //std::cout << "cd timing 3:" << duration32 << '\n';
    double * err = new double(0);
    gradient_update(n,vis0,hid0,vis,hid,dW,dc,db,err);
    boost::posix_time::ptime time_4(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration43(time_4 - time_3);
    //std::cout << "cd timing 4:" << duration43 << '\n';
    *err = sqrt(*err);
    for(int t=2;t<3&&t<errs.size();t++)
      *err += (errs[errs.size()+1-t]-*err)/t;
    errs.push_back(*err);
    test_errs.push_back(*err);
    static int cnt2 = 0;
    if(cnt2%100==0)
    std::cout << "rbm error=" << *err << std::endl;
    cnt2++;
    boost::posix_time::ptime time_5(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration54(time_5 - time_4);
    //std::cout << "cd timing 5:" << duration54 << '\n';
    //std::cout << "epsilon = " << epsilon << std::endl;
    add(W,dW,-epsilon,v*h);
    add(c,dc,-epsilon,h);
    add(b,db,-epsilon,v);

    //std::cout << "dW norm = " << norm(dW,v*h) << std::endl;
    //std::cout << "dc norm = " << norm(dc,h) << std::endl;
    //std::cout << "db norm = " << norm(db,v) << std::endl;
    //std::cout << "W norm = " << norm(W,v*h) << std::endl;
    //std::cout << "c norm = " << norm(c,h) << std::endl;
    //std::cout << "b norm = " << norm(b,v) << std::endl;
    //std::cout << "err = " << *err << std::endl;
    delete err;

    boost::posix_time::ptime time_6(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration65(time_6 - time_5);
    //std::cout << "cd timing 6:" << duration65 << '\n';
    //char ch;
    //std::cin >> ch;
  }

  void sigmoid(double * p,double * X,long n)
  {
    for(long i=0;i<n;i++)
    {
      p[i] = 1.0f/(1.0f + exp(-X[i]));
    }
  }

  void vis2hid_simple(const double * X,double * H)
  {
    {
      for(long j=0;j<h;j++)
      {
        H[j] = c[j]; 
        for(long i=0;i<v;i++)
        {
          H[j] += W[i*h+j] * X[i];
        }
        H[j] = 1.0f/(1.0f + exp(-H[j]));
      }
    }
  }

  void hid2vis_simple(const double * H,double * V)
  {
    {
      for(long i=0;i<v;i++)
      {
        V[i] = b[i]; 
        for(long j=0;j<h;j++)
        {
          V[i] += W[i*h+j] * H[j];
        }
        V[i] = 1.0f/(1.0f + exp(-V[i]));
      }
    }
  }

  void vis2hid(const double * X,double * H)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(vis2hid_worker,X,H,h,v,c,W,vrtx[thread]));
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
    threads.clear();
    vrtx.clear();
  }

  void gradient_update(long n,double * vis0,double * hid0,double * vis,double * hid,double * dW,double * dc,double * db,double * err)
  {
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());

    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
    std::vector<gradient_info*> g;

    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration10(time_1 - time_0);
    //std::cout << "gradient update timing 1:" << duration10 << '\n';

    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration21(time_2 - time_1);
    //std::cout << "gradient update timing 2:" << duration21 << '\n';
    for(long i=0;i<vrtx.size();i++)
    {
      g.push_back(new gradient_info());
    }
    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration32(time_3 - time_2);
    //std::cout << "gradient update timing 3:" << duration32 << '\n';
    for(long thread=0;thread<vrtx.size();thread++)
    {
      g[thread]->n = n;
      g[thread]->v = v;
      g[thread]->h = h;
      g[thread]->vis0 = vis0;
      g[thread]->hid0 = hid0;
      g[thread]->vis = vis;
      g[thread]->hid = hid;
      g[thread]->dW = dW;
      g[thread]->dc = dc;
      g[thread]->db = db;
      g[thread]->init();
      threads.push_back(new boost::thread(gradient_worker,g[thread],vrtx[thread]));
    }
    boost::posix_time::ptime time_4(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration43(time_4 - time_3);
    //std::cout << "gradient update timing 4:" << duration43 << '\n';
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
      g[thread]->globalUpdate();
      *err += g[thread]->partial_err;
      g[thread]->destroy();
      delete g[thread];
    }
    boost::posix_time::ptime time_5(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration54(time_5 - time_4);
    //std::cout << "gradient update timing 5:" << duration54 << '\n';
    threads.clear();
    vrtx.clear();
    g.clear();
  }
  
  void hid2vis(const double * H,double * V)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(hid2vis_worker,H,V,h,v,b,W,vrtx[thread]));
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
    threads.clear();
    vrtx.clear();
  }

};

struct DataUnit
{
  DataUnit *   hidden;
  DataUnit *  visible;
  DataUnit * visible0;
  long h,v;
  double * W;
  double * b;
  double * c;
  RBM * rbm;
  long num_iters;
  long batch_iter;
  DataUnit(long _v,long _h,long _num_iters = 100,long _batch_iter = 1)
  {
    num_iters = _num_iters;
    batch_iter = _batch_iter;
    v = _v;
    h = _h;
    W = new double[v*h];
    b = new double[v];
    c = new double[h];
    constant(c,0.5f,h);
    constant(b,0.5f,v);
    constant(W,0.5f,v*h);
      hidden = NULL;
     visible = NULL;
    visible0 = NULL;
  }

  void train(double * dat, long n, long total_n,int n_cd,double epsilon,long n_var)
  {
    // RBM(long _v,long _h,double * _W,double * _b,double * _c,long _n,double * _X)
    rbm = new RBM(v,h,W,b,c,n,dat);
    for(long i=0;i<num_iters;i++)
    {
      //std::cout << "DataUnit::train i=" << i << std::endl;
      long offset = (rand()%(total_n-n));
      for(long k=0;k<batch_iter;k++)
      {
        rbm->init(offset);
        //std::cout << "prog:" << 100*(double)k/batch_iter << "%" << std::endl;
        rbm->cd(n_cd,epsilon,offset*n_var);
      }
    }
    //char ch;
    //std::cin >> ch;
  }

  void transform(double* X,double* Y)
  {
    rbm->vis2hid(X,Y);
  }

  void initialize_weights(DataUnit* d)
  {
    if(v==d->h&&h==d->v)
    {
      for(int i=0;i<v;i++)
      {
        for(int j=0;j<h;j++)
        {
          W[i*h+j] = d->W[j*d->h+i];
        }
      }
    }
  }

  void initialize_weights(DataUnit* d1,DataUnit* d2)
  {
    if(v==d1->h+d2->h&&d1->v==h&&d2->v==h)
    {
      std::cout << "initialize bottleneck" << std::endl;
      //char ch;
      //std::cin >> ch;
      int j=0;
      for(int k=0;j<d1->h;j++,k++)
      {
        for(int i=0;i<d1->v;i++)
        {
          W[i*h+j] = d1->W[k*d1->h+i];
        }
      }
      for(int k=0;j<d1->h+d2->h;j++,k++)
      {
        for(int i=0;i<d2->v;i++)
        {
          W[i*h+j] = d2->W[k*d2->h+i];
        }
      }
    }
  }

};

// Multi Layer RBM
//
//  Auto-encoder
//
//          [***]
//         /     \
//     [*****] [*****]
//       /         \
// [********]   [********]
//   inputs      outputs
//
struct mRBM
{
  long in_samp;
  long out_samp;
  bool model_ready;
  std::vector<DataUnit*>  input_branch;
  std::vector<DataUnit*> output_branch;
  DataUnit* bottle_neck;
  void addInputDatUnit(long v,long h)
  {
    DataUnit * unit = new DataUnit(v,h);
    input_branch.push_back(unit);
  }
  void addOutputDatUnit(long v,long h)
  {
    output_branch.push_back(new DataUnit(v,h));
  }
  void addBottleNeckDatUnit(long v,long h)
  {
    bottle_neck = new DataUnit(v,h);
  }
  void construct(std::vector<long> input_num,std::vector<long> output_num,long bottle_neck_num)
  {
    for(long i=0;i+1<input_num.size();i++)
    {
      input_branch.push_back(new DataUnit(input_num[i],input_num[i+1]));
    }
    for(long i=0;i+1<output_num.size();i++)
    {
      output_branch.push_back(new DataUnit(output_num[i],output_num[i+1]));
    }
    bottle_neck = new DataUnit(input_num[input_num.size()-1]+output_num[output_num.size()-1],bottle_neck_num);
  }
  mRBM(long _in_samp,long _out_samp)
  {
    in_samp = _in_samp;
    out_samp = _out_samp;
    model_ready = false;
    bottle_neck = NULL;
  }
  void copy(double * X,double * Y,long num)
  {
    for(long i=0;i<num;i++)
    {
      Y[i] = X[i];
    }
  }
  void model_simple(long sample,double * in,double * out)
  {
    double * X = NULL;
    double * Y = NULL;
    X = new double[input_branch[0]->v];
    for(long i=0;i<input_branch[0]->v;i++)
    {
      X[i] = in[sample*input_branch[0]->v+i];
    }
    for(long i=0;i<input_branch.size();i++)
    {
      Y = new double[input_branch[i]->h];
      input_branch[i]->rbm->vis2hid_simple(X,Y);
      delete [] X;
      X = NULL;
      X = new double[input_branch[i]->h];
      copy(Y,X,input_branch[i]->h);
      delete [] Y;
      Y = NULL;
    }
    double * X_bottleneck = NULL;
    X_bottleneck = new double[bottle_neck->h];
    for(long i=0;i<input_branch[input_branch.size()-1]->h;i++)
    {
      X_bottleneck[i] = X[i];
    }
    for(long i=input_branch[input_branch.size()-1]->h;i<bottle_neck->h;i++)
    {
      X_bottleneck[i] = 0;
    }
    delete [] X;
    X = NULL;
    {
      double * Y_bottleneck = NULL;
      Y_bottleneck = new double[bottle_neck->h];
      bottle_neck->rbm->vis2hid_simple(X_bottleneck,Y_bottleneck);
      bottle_neck->rbm->hid2vis_simple(Y_bottleneck,X_bottleneck);
      delete [] Y_bottleneck;
      Y_bottleneck = NULL;
      Y = new double[out_samp];
      for(long i=input_branch[input_branch.size()-1]->h,k=0;i<bottle_neck->h;i++,k++)
      {
        Y[k] = X_bottleneck[i];
      }
      delete [] X_bottleneck;
      X_bottleneck = NULL;
    }
    for(long j=0;j<out_samp;j++)
    {
      out[sample*out_samp+j] = Y[j];//(Y[j]+1e-5)/(Y_max+1e-5);
    }
    for(long i=output_branch.size()-1;i>=0;i--)
    {
      X = new double[output_branch[i]->v];
      output_branch[i]->rbm->hid2vis_simple(Y,X);
      delete [] Y;
      Y = NULL;
      Y = new double[output_branch[i]->v];
      copy(X,Y,output_branch[i]->v);
      delete [] X;
      X = NULL;
      for(long j=0;j<output_branch[i]->v;j++)
      {
        out[sample*output_branch[i]->v+j] = Y[j];
      }
    }
    delete [] Y;
    Y = NULL;
  }
  double ** model(long sample,double * in)
  {
    double ** out = new double*[20];
    for(int i=0;i<20;i++)out[i]=new double[bottle_neck->v];
    for(int l=0;l<20;l++)
    for(int i=0;i<bottle_neck->v;i++)
    out[l][i]=0;
    //std::cout << "model:\t\t";
    //for(int i=0;i<input_branch[0]->v;i++)
    //std::cout << in[sample*input_branch[0]->v+i] << '\t';
    //std::cout << '\n';
    long layer = 0;
    double * X = NULL;
    double * Y = NULL;
    X = new double[input_branch[0]->v];
    for(long i=0;i<input_branch[0]->v;i++)
    {
      X[i] = in[sample*input_branch[0]->v+i];
    }
    for(long i=0;i<input_branch[0]->v;i++)
    {
      out[layer][i] = X[i];
    }
    //std::cout << "out:\t\t";
    //for(int i=0;i<input_branch[0]->v;i++)
    //std::cout << out[layer][i] << '\t';
    //std::cout << '\n';
    layer++;
    //std::cout << "input_branch size:" << input_branch.size() << std::endl;
    for(long i=0;i<input_branch.size();i++)
    {
      Y = new double[input_branch[i]->h];
      input_branch[i]->rbm->vis2hid_simple(X,Y);
      delete [] X;
      X = NULL;
      X = new double[input_branch[i]->h];
      copy(Y,X,input_branch[i]->h);
      delete [] Y;
      Y = NULL;
      for(long j=0;j<input_branch[i]->h;j++)
      {
        out[layer][j] = X[j];
      }
      layer++;
    }
    double * X_bottleneck = NULL;
    X_bottleneck = new double[bottle_neck->h];
    for(long i=0;i<input_branch[input_branch.size()-1]->h;i++)
    {
      X_bottleneck[i] = X[i];
    }
    for(long i=input_branch[input_branch.size()-1]->h;i<bottle_neck->h;i++)
    {
      X_bottleneck[i] = 0;
    }
    delete [] X;
    X = NULL;
    {
      double * Y_bottleneck = NULL;
      Y_bottleneck = new double[bottle_neck->h];
      bottle_neck->rbm->vis2hid_simple(X_bottleneck,Y_bottleneck);
      for(long j=0;j<bottle_neck->v;j++)
      {
        out[layer][j] = X_bottleneck[j];
      }
      layer++;
      for(long j=0;j<bottle_neck->v;j++)
      {
        out[layer][j] = Y_bottleneck[j];
      }
      layer++;
      bottle_neck->rbm->hid2vis_simple(Y_bottleneck,X_bottleneck);
      for(long j=0;j<bottle_neck->v;j++)
      {
        out[layer][j] = X_bottleneck[j];
      }
      layer++;
      delete [] Y_bottleneck;
      Y_bottleneck = NULL;
      Y = new double[out_samp];
      for(long i=input_branch[input_branch.size()-1]->h,k=0;i<bottle_neck->h;i++,k++)
      {
        Y[k] = X_bottleneck[i];
      }
      delete [] X_bottleneck;
      X_bottleneck = NULL;
    }
    //double Y_max = 0;
    //for(long j=0;j<bottle_neck->v-input_branch[input_branch.size()-1]->v;j++)
    //{
    //  if(Y[j]>Y_max)Y_max = Y[j];
    //}
    for(long j=0;j<out_samp;j++)
    {
      out[layer][j] = Y[j];//(Y[j]+1e-5)/(Y_max+1e-5);
    }
    layer++;
    for(long i=output_branch.size()-1;i>=0;i--)
    {
      X = new double[output_branch[i]->v];
      output_branch[i]->rbm->hid2vis_simple(Y,X);
      delete [] Y;
      Y = NULL;
      Y = new double[output_branch[i]->v];
      copy(X,Y,output_branch[i]->v);
      delete [] X;
      X = NULL;
      for(long j=0;j<output_branch[i]->v;j++)
      {
        out[layer][j] = Y[j];
      }
      layer++;
    }
    //for(long i=0;i<output_branch[0]->v;i++)
    //{
    //  out[layer][i] = Y[i];
    //}
    delete [] Y;
    Y = NULL;
    return out;
  }
  void train(long in_num,long out_num,long n_samp,long total_n,long n_cd,double epsilon,double * in,double * out)
  {
    double * X = NULL;
    double * Y = NULL;
    double * IN = NULL;
    double * OUT = NULL;
    X = new double[in_num*n_samp];
    IN = new double[in_num*n_samp];
    for(long i=0;i<in_num*n_samp;i++)
    {
      X[i] = in[i];
    }
    for(long i=0;i<input_branch.size();i++)
    {
      if(i>0)input_branch[i]->initialize_weights(input_branch[i-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      input_branch[i]->train(X,n_samp,total_n,n_cd,epsilon,input_branch[i]->h);
      Y = new double[input_branch[i]->h*n_samp];
      input_branch[i]->transform(X,Y);
      delete [] X;
      X = NULL;
      //std::cout << "X init:" << in_num*n_samp << "    " << "X fin:" << input_branch[i]->h*n_samp << std::endl;
      X = new double[input_branch[i]->h*n_samp];
      copy(Y,X,input_branch[i]->h*n_samp);
      copy(Y,IN,input_branch[i]->h*n_samp);
      delete [] Y;
      Y = NULL;
    }
    delete [] X;
    X = NULL;
    X = new double[out_num*n_samp];
    OUT = new double[in_num*n_samp];
    for(long i=0;i<out_num*n_samp;i++)
    {
      X[i] = out[i];
      OUT[i] = out[i];
    }
    for(long i=0;i<output_branch.size();i++)
    {
      if(i>0)output_branch[i]->initialize_weights(output_branch[i-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      output_branch[i]->train(X,n_samp,total_n,n_cd,epsilon,input_branch[i]->h);
      Y = new double[output_branch[i]->h*n_samp];
      output_branch[i]->transform(X,Y);
      delete [] X;
      X = NULL;
      X = new double[output_branch[i]->h*n_samp];
      copy(Y,X,output_branch[i]->h*n_samp);
      copy(Y,OUT,output_branch[i]->h*n_samp);
      delete [] Y;
      Y = NULL;
    }
    delete [] X;
    X = NULL;
    if(bottle_neck!=NULL)
    {
      X = new double[bottle_neck->h*n_samp];
      for(long s=0;s<n_samp;s++)
      {
        long i=0;
        for(long k=0;i<in_num&&k<in_num;i++,k++)
        {
          X[s*(in_num+out_num)+i] = IN[s*in_num+k];
        }
        for(long k=0;i<in_num+out_num&&k<out_num;i++,k++)
        {
          X[s*(in_num+out_num)+i] = OUT[s*out_num+k];
        }
      }
      //bottle_neck->initialize_weights(input_branch[input_branch.size()-1],output_branch[output_branch.size()-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      bottle_neck->train(X,n_samp,total_n,n_cd,epsilon,in_num+out_num);
      delete [] X;
      X = NULL;
    }
    delete [] IN;
    IN = NULL;
    delete [] OUT;
    OUT = NULL;
    model_ready = true;
  }
  double compare(long sample,double * a,double * b)
  {
    double sum_a = 0;
    double sum_b = 0;
    for(int i=0;i<out_samp;i++)
    {
      sum_a += a[sample*out_samp+i];
      sum_b += b[sample*out_samp+i];
    }
    for(int i=0;i<out_samp;i++)
    {
      a[sample*out_samp+i] = (a[sample*out_samp+i]+1e-5)/(sum_a+1e-5);
      b[sample*out_samp+i] = (b[sample*out_samp+i]+1e-5)/(sum_b+1e-5);
    }
    double score = 0;
    for(int i=0;i<out_samp;i++)
    {
      score += ((a[sample*out_samp+i]>0.5&&b[sample*out_samp+i]>0.5)||(a[sample*out_samp+i]<0.5&&b[sample*out_samp+i]<0.5))?1:0;
    }
    return score/out_samp;
  }
  void compare_all(long num,double * in,double * out)
  {
    double score = 0;
    for(long i=0;i<num;i++)
    {
      score += (compare(i,in,out)-score)/(1+i);
      for(int j=0;j<out_samp;j++)
      {
        std::cout << ((in[i*out_samp+j]>0.5)?"1":"0") << ":" << ((out[i*out_samp+j]>0.5)?"1":"0") << "\t";
      }
    }
    std::cout << std::endl;
    for(long i=0;i<num;i++)
    {
      for(int j=0;j<out_samp;j++)
      {
        std::cout << in[i*out_samp+j] << ":" << out[i*out_samp+j] << "\t";
      }
    }
    std::cout << std::endl;
    std::cout << "score:" << score << std::endl;
    //char ch;
    //std::cin >> ch;
  }
  void model_all(long num,double * in,double * out)
  {
    for(long i=0;i<num;i++)
    {
      model_simple(i,in,out);
    }
  }
};

mRBM * mrbm = NULL;

/*
 
 RSI Oversold in Uptrend

 This scan reveals stocks that are in an uptrend with oversold RSI. First, stocks must be above their 200-day moving average to be in an overall uptrend. Second, RSI must cross below 30 to become oversold.

 [type = stock] AND [country = US] 
 AND [Daily SMA(20,Daily Volume) > 40000] 
 AND [Daily SMA(60,Daily Close) > 20] 

 AND [Daily Close > Daily SMA(200,Daily Close)] 
 AND [Daily RSI(5,Daily Close) <= 30]

 RSI Overbought in Downtrend

 This scan reveals stocks that are in a downtrend with overbought RSI turning down. First, stocks must be below their 200-day moving average to be in an overall downtrend. Second, RSI must cross above 70 to become overbought.

 [type = stock] AND [country = US] 
 AND [Daily SMA(20,Daily Volume) > 40000] 
 AND [Daily SMA(60,Daily Close) > 20] 

 AND [Daily Close < Daily SMA(200,Daily Close)] 
 AND [Daily RSI(5,Daily Close) >= 70]

*/

double max(double a,double b)
{
  return (a>b)?a:b;
}

double min(double a,double b)
{
  return (a>b)?b:a;
}

double fabs(double a)
{
  return (a>0)?a:-a;
}

struct Point
{
  double t;
  double x,y;
  Point(double _x,double _y):x(_x),y(_y){}
  double dist(Point const & a,double alpha)
  {
    return pow(sqrt((x-a.x)*(x-a.x) + (y-a.y)*(y-a.y)),alpha);
  }
};

double total_dist(std::vector<Point> & pts,double alpha = 1)
{
  double temp_dist;
  double total_dist = 0;
  pts[0].t = total_dist;
  for(int i=1;i<pts.size();i++)
  {
    temp_dist = pts[i].dist(pts[i-1],alpha);
    total_dist += temp_dist;
    pts[i].t = total_dist;
  }
}

int find(std::vector<Point> & pts,double t)
{
  for(int i=1;i<pts.size();i++)
  {
    if(pts[i].t > t)return i-1;
  }
}

Point CatmulRom(std::vector<Point> & pts,double t)
{
  int ind0 = (int)find(pts,t)-1;
  int ind1 = (int)find(pts,t);
  int ind2 = (int)find(pts,t)+1;
  int ind3 = (int)find(pts,t)+2;
  //std::cout << pts[ind1].t << "\t" << pts[ind1].x << "\t" << pts[ind1].y << std::endl;
  //std::cout << "$$$$$" << ind1 << std::endl;
  if(ind0<0)ind0 = 0;
  if(ind1<0)ind1 = 0;
  if(ind2<0)ind2 = 0;
  if(ind3<0)ind3 = 0;
  if(ind0>=pts.size())ind0 = pts.size()-1;
  if(ind1>=pts.size())ind1 = pts.size()-1;
  if(ind2>=pts.size())ind2 = pts.size()-1;
  if(ind3>=pts.size())ind3 = pts.size()-1;
  //std::cout << t << "~~~~" << pts[ind0].t << '\t' << pts[ind1].t << '\t' << pts[ind2].t << '\t' << pts[ind3].t << std::endl;
  //std::cout << "^^^^" << pts[ind0].x << '\t' << pts[ind1].x << '\t' << pts[ind2].x << '\t' << pts[ind3].x << std::endl;
  double d10 = pts[ind1].t - pts[ind0].t;
  double d21 = pts[ind2].t - pts[ind1].t;
  double d32 = pts[ind3].t - pts[ind2].t;
  double A1x = (d10>1e-5)?(pts[ind0].x*(pts[ind1].t-t) + pts[ind1].x*(t-pts[ind0].t))/d10:pts[ind0].x;
  double A1y = (d10>1e-5)?(pts[ind0].y*(pts[ind1].t-t) + pts[ind1].y*(t-pts[ind0].t))/d10:pts[ind0].y;
  double A2x = (d21>1e-5)?(pts[ind1].x*(pts[ind2].t-t) + pts[ind2].x*(t-pts[ind1].t))/d21:pts[ind1].x;
  double A2y = (d21>1e-5)?(pts[ind1].y*(pts[ind2].t-t) + pts[ind2].y*(t-pts[ind1].t))/d21:pts[ind1].y;
  double A3x = (d32>1e-5)?(pts[ind2].x*(pts[ind3].t-t) + pts[ind3].x*(t-pts[ind2].t))/d32:pts[ind2].x;
  double A3y = (d32>1e-5)?(pts[ind2].y*(pts[ind3].t-t) + pts[ind3].y*(t-pts[ind2].t))/d32:pts[ind2].y;
  //std::cout << "^^^" << A1x << '\t' << A2x << '\t' << A3x << std::endl;
  double d20 = pts[ind2].t - pts[ind0].t;
  double d31 = pts[ind3].t - pts[ind1].t;
  double B1x = (d20>1e-5)?(A1x*(pts[ind2].t-t) + A2x*(t-pts[ind0].t))/d20:A1x;
  double B1y = (d20>1e-5)?(A1y*(pts[ind2].t-t) + A2y*(t-pts[ind0].t))/d20:A1y;
  double B2x = (d31>1e-5)?(A2x*(pts[ind3].t-t) + A3x*(t-pts[ind1].t))/d31:A2x;
  double B2y = (d31>1e-5)?(A2y*(pts[ind3].t-t) + A3y*(t-pts[ind1].t))/d31:A2y;
  //std::cout << "^^" << B1x << '\t' << B2x << std::endl;
  double Cx  = (d21>1e-5)?(B1x*(pts[ind2].t-t) + B2x*(t-pts[ind1].t))/d21:B1x;
  double Cy  = (d21>1e-5)?(B1y*(pts[ind2].t-t) + B2y*(t-pts[ind1].t))/d21:B1y;
  //std::cout << "^" << Cx << "\t" << Cy << std::endl;
  return Point(Cx,Cy);
}


Point estimate_derivative(std::vector<Point> & pts,double a,double dx)
{
  Point p1 = CatmulRom(pts,a-dx);
  Point p2 = CatmulRom(pts,a+dx);
  return Point((p2.x-p1.x)/(2*dx),(p2.y-p1.y)/(2*dx));
}

struct price
{

  price()
  {
    prediction_confidence = 0;
    synthetic = false;
    EMA = 0;
    EMA1 = 0;
    EMS = 0;
    EMS1 = 0;
  }

  // these quantities are ground truth
  bool synthetic;
  int index;
  std::string date;
  double open;
  double close;
  double high;
  double low;
  int volume;
  double prev_close;
  double prct_change;
  double prct_prediction;
  double close_prediction;
  double auto_encoding_x;
  double auto_encoding_y;
  double prediction_confidence;

  // these quantities are derived from the values above
  
  double EMA_MACD; // temporary ema MACD
  double calculate_ema_macd(std::vector<price> & prices,int N)
  {
    double a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMA_MACD = a * (MACD_line - prices[index-1].EMA_MACD) + prices[index-1].EMA_MACD;
      return EMA_MACD;
    }
    else
    {
      EMA_MACD = MACD_line;
      return EMA_MACD;
    }
  }
  double EMA_MACD1; // temporary ema MACD
  double calculate_ema_macd1(std::vector<price> & prices,int N)
  {
    double a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMA_MACD1 = a * (MACD_line - prices[index-1].EMA_MACD1) + prices[index-1].EMA_MACD1;
      return EMA_MACD1;
    }
    else
    {
      EMA_MACD1 = MACD_line;
      return EMA_MACD1;
    }
  }
  double EMA; // temporary ema
  double calculate_ema(std::vector<price> & prices,int N)
  {
    double a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMA = a * (close - prices[index-1].EMA) + prices[index-1].EMA;
      return EMA;
    }
    else
    {
      EMA = close;
      return EMA;
    }
  }
  double EMA1; // temporary ema
  double calculate_ema1(std::vector<price> & prices,int N)
  {
    double a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMA1 = a * (close - prices[index-1].EMA1) + prices[index-1].EMA1;
      return EMA1;
    }
    else
    {
      EMA1 = close;
      return EMA1;
    }
  }
  double EMS; // temporary ems
  double calculate_ems(std::vector<price> & prices,double mean,int N)
  {
    double a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMS = a * ((close - mean)*(close - mean) - prices[index-1].EMS) + prices[index-1].EMS;
      return EMS;
    }
    else
    {
      EMS = 0;
      return EMS;
    }
  }
  double EMS1; // temporary ems
  double calculate_ems1(std::vector<price> & prices,double mean,int N)
  {
    double a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMS1 = a * ((close - mean)*(close - mean) - prices[index-1].EMS1) + prices[index-1].EMS1;
      return EMS1;
    }
    else
    {
      EMS1 = 0;
      return EMS1;
    }
  }
  double EMAV; // temporary ema
  double calculate_ema_volume(std::vector<price> & prices,int N)
  {
    double a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMAV = a * (volume - prices[index-1].EMAV) + prices[index-1].EMAV;
      return EMAV;
    }
    else
    {
      EMAV = volume;
      return EMAV;
    }
  }
  double SMAV; // temporary simple moving average
  double calculate_sma_volume(std::vector<price> & prices,int N)
  {
    if(index>=N)
    {
      SMAV = 0;
      for(int i=1;i<=N;i++)
      {
        SMAV += (prices[index+1-i].volume - SMAV)/i;
      }
      return SMAV;
    }
    else
    {
      SMAV = 0;
      for(int i=1;i<=index+1;i++)
      {
        SMAV += (prices[index+1-i].volume - SMAV)/i;
      }
      return SMAV;
    }
  }

  // Volume Spike - sizzle index
  // Current Volume > tolerance x (5 d SMA Volume)
  // close today > close yesterday => volume gain
  // close today < close yesterday => volume loss
  bool Volume_spike;
  bool Volume_spike_gain;
  bool Volume_spike_loss;
  bool Volume_gain;
  bool Volume_loss;
  void calculate_Volume_spike(std::vector<price> & prices,int N=5,double tolerance=2.0f)
  {
    Volume_spike = volume > tolerance * SMAV;
    Volume_spike_gain = false;
    Volume_spike_loss = false;
    Volume_gain = false;
    Volume_loss = false;
    if(index >= 1)
    {
      if(Volume_spike)
      {
        if(close > prices[index-1].close)
        {
          Volume_spike_gain = true;
        }
        else
        {
          Volume_spike_loss = true;
        }
      }
      if(close > prices[index-1].close)
      {
        Volume_gain = true;
      }
      else
      {
        Volume_loss = true;
      }
    }
  }
  static void initialize_Volume_spike(std::vector<price> & prices, int N=5,double tolerance=2.0f)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_sma_volume(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_Volume_spike(prices,N,tolerance);
    }
  }

  // RSI 
  // 100 - 100/(1+RS)
  // RS = Average Gain / Average Loss
  double average_gain;
  double average_loss;
  double RS;
  double RSI;
  bool RSI_buy;
  bool RSI_sell;
  void calculate_RSI(std::vector<price> & prices,int N=14)
  {
    if(index==N)
    {
      average_gain = 0;
      average_loss = 0;
      for(int i=1;i<N+1;i++)
      {
        if(prices[index+1-i].close > prices[index-i].close)
        {
          average_gain += ((prices[index+1-i].close-prices[index-i].close) - average_gain)/i;
          average_loss -= average_loss/i;
        }
        else
        {
          average_loss += ((prices[index-i].close-prices[index+1-i].close) - average_loss)/i;
          average_gain -= average_gain/i;
        }
      }
      if(average_loss>1e-10)
      {
        RS = average_gain / average_loss;
      }
      else
      {
        RS = 0;
      }
      RSI = 100 - 100/(1+RS);
    }
    else
    if(index>=N+1)
    {
      if(prices[index].close > prices[index-1].close)
      {
        average_gain = ((prices[index].close-prices[index-1].close) + (N-1)*prices[index-1].average_gain)/N;
        average_loss = ((N-1)*prices[index-1].average_loss)/N;
      }
      else
      {
        average_loss = ((prices[index-1].close-prices[index].close) + (N-1)*prices[index-1].average_loss)/N;
        average_gain = ((N-1)*prices[index-1].average_gain)/N;
      }
      if(average_loss>1e-10)
      {
        RS = average_gain / average_loss;
      }
      else
      {
        RS = 0;
      }
      RSI = 100 - 100/(1+RS);
    }
    else
    {
      average_gain = 0;
      average_loss = 0;
      RS = 0;
      RSI = 0;
    }
    //std::cout << index << "\t" << date << "\t" << average_gain << "\t" << average_loss << "\t" << RS << "\t" << RSI << std::endl;
    //char ch;
    //std::cin >> ch;
  }
  void calculate_RSI_buy()
  {
    if(RSI>1e-10)
    {
      RSI_buy = RSI<30;
    }
    else
    {
      RSI_buy = false;
    }
  }
  void calculate_RSI_sell()
  {
    if(RSI>1e-10)
    {
      RSI_sell = RSI>70;
    }
    else
    {
      RSI_sell = false;
    }
  }
  static void initialize_RSI(std::vector<price> & prices,int N=14)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_RSI(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_RSI_buy();
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_RSI_sell();
    }
  }

  // MFI 
  // 100 - 100/(1+MF)
  // MF = Average TP Gain * Gain Volume / Average TP Loss * Loss Volume
  double raw_money_flow_gain;
  double raw_money_flow_loss;
  double MF;
  double MFI;
  bool MFI_buy;
  bool MFI_sell;
  void calculate_MFI(std::vector<price> & prices,int N=14)
  {
    if(index==N)
    {
      raw_money_flow_gain = 0;
      raw_money_flow_loss = 0;
      for(int i=1;i<N+1;i++)
      {
        if(prices[index+1-i].TP > prices[index-i].TP)
        {
          raw_money_flow_gain += (prices[index+1-i].TP*prices[index+1-i].volume - raw_money_flow_gain)/i;
          raw_money_flow_loss -= raw_money_flow_loss/i;
        }
        else
        {
          raw_money_flow_loss += (prices[index+1-i].TP*prices[index+1-i].volume - raw_money_flow_loss)/i;
          raw_money_flow_gain -= raw_money_flow_gain/i;
        }
      }
      if(raw_money_flow_loss>1e-10)
      {
        MF = raw_money_flow_gain / raw_money_flow_loss;
      }
      else
      {
        MF = 0;
      }
      MFI = 100 - 100/(1+MF);
    }
    else
    if(index>=N+1)
    {
      if(prices[index].TP > prices[index-1].TP)
      {
        raw_money_flow_gain = (prices[index].TP*prices[index-1].volume + (N-1)*prices[index-1].raw_money_flow_gain)/N;
        raw_money_flow_loss = ((N-1)*prices[index-1].raw_money_flow_loss)/N;
      }
      else
      {
        raw_money_flow_loss = (prices[index-1].TP*prices[index].volume + (N-1)*prices[index-1].raw_money_flow_loss)/N;
        raw_money_flow_gain = ((N-1)*prices[index-1].raw_money_flow_gain)/N;
      }
      if(raw_money_flow_loss>1e-10)
      {
        MF = raw_money_flow_gain / raw_money_flow_loss;
      }
      else
      {
        MF = 0;
      }
      MFI = 100 - 100/(1+MF);
    }
    else
    {
      raw_money_flow_gain = 0;
      raw_money_flow_loss = 0;
      MF = 0;
      MFI = 0;
    }
  }
  void calculate_MFI_buy()
  {
    if(MFI>1e-10)
    {
      MFI_buy = MFI<30;
    }
    else
    {
      MFI_buy = false;
    }
  }
  void calculate_MFI_sell()
  {
    if(MFI>1e-10)
    {
      MFI_sell = MFI>70;
    }
    else
    {
      MFI_sell = false;
    }
  }
  static void initialize_MFI(std::vector<price> & prices,int N=14)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_TP();
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MFI(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MFI_buy();
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MFI_sell();
    }
  }

  // Golden cross
  double ema_50;
  double ema_200;
  double ems_50;
  double ems_200;
  bool GoldenCross_uptrend;
  bool GoldenCross_downtrend;
  void calculate_GoldenCross_uptrend(std::vector<price> & prices,int N1=50,int N2=200)
  {
    if(index>=N2&&index>=N1)
    {
      ema_50 = calculate_ema(prices,N1);
      ema_200= calculate_ema1(prices,N2);
      ems_50 = sqrtf(calculate_ems(prices,ema_50,N1));
      ems_200= sqrtf(calculate_ems1(prices,ema_200,N2));
      GoldenCross_uptrend = ema_50>ema_200;
    }
    else
    {
      ema_50 = close;
      ema_200 = close;
      ems_50 = 0;
      ems_200 = 0;
      GoldenCross_uptrend = false;
    }
  }
  void calculate_GoldenCross_downtrend(std::vector<price> & prices,int N1=50,int N2=200)
  {
    if(index>=N2&&index>=N1)
    {
      ema_50 = calculate_ema(prices,N1);
      ema_200= calculate_ema1(prices,N2);
      ems_50 = sqrtf(calculate_ems(prices,ema_50,N1));
      ems_200= sqrtf(calculate_ems1(prices,ema_200,N2));
      GoldenCross_downtrend = ema_50<ema_200;
    }
    else
    {
      ema_50 = close;
      ema_200 = close;
      ems_50 = 0;
      ems_200 = 0;
      GoldenCross_downtrend = false;
    }
  }
  static void initialize_GoldenCross(std::vector<price> & prices,int N1=50,int N2=200)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_GoldenCross_uptrend(prices,N1,N2);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_GoldenCross_downtrend(prices,N1,N2);
    }
  }

  // MACD
  double ema_12;
  double ema_26;
  double ems_12;
  double ems_26;
  double MACD_line;
  double MACD_signal;
  double MACD_dline;
  double MACD_dsignal;
  bool MACD_uptrend;
  bool MACD_downtrend;
  void calculate_MACD_signal(std::vector<price> & prices,int N=9,int N1=12,int N2=26)
  {
    if(index>=N&&index>=N1&&index>=N2&&index>=100)
    {
      ema_12 = calculate_ema(prices,N1);
      ema_26 = calculate_ema1(prices,N2);
      if(index>=200)
      {
        MACD_line = (ema_12 - ema_26)/(ema_26+1);
        MACD_signal = calculate_ema_macd(prices,N);
      }
      else
      {
        MACD_line = 0;
        MACD_signal = 0;
      }
    }
    else
    {
      ema_12 = close;
      ema_26 = close;
      MACD_line = 0;
      MACD_signal = 0;
    }
  }
  static void calculate_MACD_dsignal(std::vector<price> & prices)
  {
    //char ch;
    //std::cout << "pass 1" << std::endl;
    std::vector<Point> pts;
    for(int i=0;i<prices.size();i++)
    {
      pts.push_back(Point(1000*prices[i].MACD_line,1000*prices[i].MACD_signal));
      //std::cout << pts[i].x << "\t" << pts[i].y << std::endl;
    }
    //std::cin >> ch;
    //std::cout << "pass 2" << std::endl;
    total_dist(pts,0.5);
    //total_dist(pts,0.0);
    //for(int i=0;i<prices.size();i++)
    //{
      //std::cout << pts[i].t << "\t" << pts[i].x << "\t" << pts[i].y << std::endl;
    //}
    //std::cout << "pass 3" << std::endl;
    //std::cin >> ch;
    prices[0].MACD_dline   = 0.0001;
    prices[0].MACD_dsignal = 0.0001;
    for(int i=0;i+1<pts.size();i++)
    {
      //std::cout << "i=" << i << "\t" << pts[i].t << std::endl;
      //Point pt = CatmulRom(pts,0.5f*(pts[i].t+pts[i+1].t));
      //Point drv = estimate_derivative(pts,0.5f*(pts[i].t+pts[i+1].t),0.01);
      Point drv = estimate_derivative(pts,pts[i].t,0.01);
      prices[i+1].MACD_dline = drv.x;
      prices[i+1].MACD_dsignal = drv.y;
      //prices[i+1].MACD_dline = pt.x;
      //prices[i+1].MACD_dsignal = pt.y;
      //prices[i+1].MACD_dline = pts[i].x;
      //prices[i+1].MACD_dsignal = pts[i].y;
    }
    //std::cin >> ch;
    //std::cout << "pass 4" << std::endl;
  }
  void calculate_MACD_uptrend(std::vector<price> & prices,int N1=12,int N2=26)
  {
    if(index>=N2&&index>=N1&&index>=100)
    {
      ema_12 = calculate_ema(prices,N1);
      ema_26 = calculate_ema1(prices,N2);
      ems_12 = sqrtf(calculate_ems(prices,ema_12,N1));
      ems_26 = sqrtf(calculate_ems1(prices,ema_26,N2));
      MACD_uptrend = ema_12>ema_26;
    }
    else
    {
      ema_12 = close;
      ema_26 = close;
      ems_12 = 0;
      ems_26 = 0;
      MACD_uptrend = false;
    }
  }
  void calculate_MACD_downtrend(std::vector<price> & prices,int N1=12,int N2=26)
  {
    if(index>=N2&&index>=N1&&index>=100)
    {
      ema_12 = calculate_ema(prices,N1);
      ema_26 = calculate_ema1(prices,N2);
      ems_12 = sqrtf(calculate_ems(prices,ema_12,N1));
      ems_26 = sqrtf(calculate_ems1(prices,ema_26,N2));
      MACD_downtrend = ema_12<ema_26;
    }
    else
    {
      ema_12 = close;
      ema_26 = close;
      ems_12 = 0;
      ems_26 = 0;
      MACD_downtrend = false;
    }
  }
  static void initialize_MACD(std::vector<price> & prices,bool awesome_macd=false,int N=9,int N1=12,int N2=26)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MACD_uptrend(prices,N1,N2);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MACD_downtrend(prices,N1,N2);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MACD_signal(prices,N,N1,N2);
    }
    if(awesome_macd)
    {
      calculate_MACD_dsignal(prices);
    }
  }

  // DOJI - open and close price are very similar
  bool doji;
  void calculate_doji()
  {
    doji = open + 0.003*open > close && open - 0.003*open < close;
  }
  static void initialize_doji(std::vector<price> & prices)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_doji();
    }
  }

  // CCI
  double TP; // typical price = (high + low + close)/3
  double smtp_cci; // 20 day simple moving average of Typical Price (TP)
  double MD_cci; // 20 day mean deviation = sum_n |TP - smtp_n|/n
  double CCI; // (TP - 20d SMTP) / (.015 MD)
  bool CCI_buy, CCI_sell;
  static void initialize_CCI(std::vector<price> & prices,int N = 20)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_TP();
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_smtp(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MD(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_CCI(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_CCI_buy();
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_CCI_sell();
    }
  }
  void calculate_TP()
  {
    TP = (high + low + close)/3.0f;
  }
  void calculate_smtp(std::vector<price> & prices,int N = 20)
  {
    if(index>=N)
    {
      smtp_cci = 0;
      for(int k=1;k<=N;k++)
      {
        smtp_cci += (prices[index+1-k].TP - smtp_cci)/k;
      }
    }
    else
    {
      smtp_cci = 0;
    }
  }
  void calculate_MD(std::vector<price> & prices,int N = 20)
  {
    if(index>=N)
    {
      MD_cci = 0;
      for(int k=1;k<=N;k++)
      {
        MD_cci += (fabs(prices[index+1-k].TP - smtp_cci) - MD_cci)/k;
      }
    }
    else
    {
      smtp_cci = 0;
    }
  }
  void calculate_CCI(std::vector<price> & prices,int N=20)
  {
    if(index>=N)
    {
      CCI = (TP - smtp_cci) / (.015 * MD_cci);
    }
    else
    {
      CCI = 0;
    }
  }
  void calculate_CCI_buy()
  {
    if(fabs(CCI-0)>1e-10)
    {
      CCI_buy = CCI < -100;
    }
    else
    {
      CCI_buy = false;
    }
  }
  void calculate_CCI_sell()
  {
    if(fabs(CCI-0)>1e-10)
    {
      CCI_sell = CCI > 100;
    }
    else
    {
      CCI_sell = false;
    }
  }

  // Bullish Engulfing Pattern
  // Today close > Yesterday open
  // Today open < Yesterday close
  // Today close > Today open // today candlestick is green
  // Yesterday close < Yesterday open // yesterday candlestick is red
  bool bullish_engulfing_pattern;
  void calculate_bullish_engulfing_pattern(std::vector<price> & prices)
  {
    if(index>=1)
    {
      bullish_engulfing_pattern = (close>prices[index-1].open)
                                &&(open<prices[index-1].close)
                                &&(close>open)
                                &&(prices[index-1].close<prices[index-1].open)
                                ;
    }
    else
    {
      bullish_engulfing_pattern = false;
    }
  }

  // Bearish Engulfing Pattern
  // Today open > Yesterday close
  // Today close < Yesterday open
  // Today close < Today open // today candlestick is red
  // Yesterday close > Yesterday open // yesterday candlestick is green
  bool bearish_engulfing_pattern;
  void calculate_bearish_engulfing_pattern(std::vector<price> & prices)
  {
    if(index>=1)
    {
      bearish_engulfing_pattern = (open>prices[index-1].close)
                                &&(close<prices[index-1].open)
                                &&(close<open)
                                &&(prices[index-1].close>prices[index-1].open)
                                ;
    }
    else
    {
      bearish_engulfing_pattern = false;
    }
  }

  static void initialize_engulfing_patterns(std::vector<price> & prices)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_bullish_engulfing_pattern(prices);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_bearish_engulfing_pattern(prices);
    }
  }

  static void initialize_percent_change(std::vector<price> & prices)
  {
    prices[0].prev_close = prices[0].close;
    prices[0].prct_change = 0;
    prices[0].auto_encoding_x = 0;
    prices[0].auto_encoding_y = 0;
    prices[0].close_prediction = prices[0].close;
    prices[0].prct_prediction = 0;
    for(int i=1;i<prices.size();i++)
    {
      prices[i].prev_close = prices[i-1].close;
      prices[i].prct_change = (prices[i].close - prices[i-1].close) / prices[i-1].close;
      prices[i].auto_encoding_x = 0;
      prices[i].auto_encoding_y = 0;
      prices[i].close_prediction = prices[i-1].close;
      prices[i].prct_prediction = 0;
    }
  }

  // initialize all indicators 
  static void initialize_indicators(std::vector<price> & prices,bool awesome_macd)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_ema_volume(prices,500);
    }
    initialize_CCI(prices);
    initialize_doji(prices);
    initialize_MACD(prices,awesome_macd);
    initialize_GoldenCross(prices);
    initialize_engulfing_patterns(prices);
    initialize_RSI(prices);
    initialize_MFI(prices);
    initialize_Volume_spike(prices);
    initialize_percent_change(prices);
  }

};

struct Scanner
{
  std::set<std::string> buy;
  void scan(std::vector<std::vector<price> > & prices, std::vector<std::string> & symbols)
  {
    buy . clear();
    for(int i=0;i<prices.size();i++)
    {
      if(prices[i].size()>=3)
      {
        for(int d=0;d<3;d++)
        {
          if( prices[i][prices[i].size()-1-d].CCI_buy ||
              prices[i][prices[i].size()-1-d].RSI_buy ||
              prices[i][prices[i].size()-1-d].MFI_buy 
            )
          {
            buy.insert(symbols[i]);
          }
        }
      }
    }
    std::cout << "Buy candidates:" << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    for(std::set<std::string>::iterator it = buy.begin();it != buy.end();it++)
    {
      std::cout << *it << std::endl;
    }
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  }
};

bool comparator(price a,price b){return a.close < b.close;}
bool comparator_MACD(price a,price b){return a.MACD_dline < b.MACD_dline;}
bool comparator_auto(price a,price b){return a.auto_encoding_x < b.auto_encoding_x;}
bool comparator_low(price a,price b){return a.low < b.low;}
bool comparator_high(price a,price b){return a.high < b.high;}

bool comparator_volume(price a,price b){return a.volume < b.volume;}

struct Bin
{
  double price_min;
  double price_max;
  double sum;
  double sum_neg;
  double sum_pos;
  bool in(double price)
  {
    return price>=price_min&&price<price_max;
  }
  std::vector<price> collection;
  Bin(double _price_min,double _price_max)
  {
    price_min = _price_min;
    price_max = _price_max;
  }
  void push(price p)
  {
    if(in(p.close))
    {
      collection.push_back(p);
    }
  }
  void calc()
  {
    sum = 0;
    for(int i=0;i<collection.size();i++)
    {
      sum += collection[i].volume;
      if(collection[i].Volume_gain) sum_pos += collection[i].volume;
      if(collection[i].Volume_loss) sum_neg += collection[i].volume;
    }
  }
};

struct VolumeByPrice
{
  double max_sum;
  std::vector<Bin> bins;
  void create_bins(std::vector<price> & prices,int nbins,int min_index,int max_index)
  {
    double max_price = double(std::max_element(prices.begin()+min_index,prices.begin()+max_index,comparator)->close);
    double min_price = double(std::min_element(prices.begin()+min_index,prices.begin()+max_index,comparator)->close);
    double bin_size = (max_price-min_price)/nbins;
    for(int i=0;i<nbins;i++)
    {
      bins.push_back(Bin(min_price+bin_size*i,min_price+(i+1)*bin_size));
    }
    for(int i=min_index;i<=max_index;i++)
    {
      for(int k=0;k<nbins;k++)
      {
        bins[k].push(prices[i]);
      }
    }
    for(int i=0;i<nbins;i++)
    {
      bins[i].calc();
    }
    max_sum = 0;
    for(int i=0;i<nbins;i++)
    {
      if(bins[i].sum>max_sum)
      {
        max_sum = bins[i].sum;
      }
    }
    for(int i=0;i<nbins;i++)
    {
      bins[i].sum     /= max_sum+1;
      bins[i].sum_pos /= max_sum+1;
      bins[i].sum_neg /= max_sum+1;
    }
  }
};

void generate_synthetic(std::vector<price> & prices,double multiplier,int nyears)
{
  double value = (rand()%10000000)/10000.0;
  int date = 1;
  int nweeks = (int)(nyears*52.1429);
  for(int w=0;w<nweeks;w++)
  {
    for(int d=0;d<5;d++)
    {
      price p;
      std::stringstream date_ss;
      date_ss << date;
      p.index = prices.size();
      p.date = date_ss.str();
      p.open = value;
      p.close = value * ( 1.0 + multiplier*0.02 * (1-2*(rand()%10000)/10000.0) );
      p.high = max(p.open,p.close) * ( 1 + multiplier*0.02 * ((rand()%10000)/10000.0) );
      p.low = min(p.open,p.close) * ( 1 - multiplier*0.02 * ((rand()%10000)/10000.0) );
      p.volume = (int)(100*fabs(p.close-p.open));
      prices.push_back(p);
      if((rand()%10000)/10000.0 > 1.0/4.0)
      {
        value *= 1.0 + multiplier*0.01 * ((rand()%10000)/10000.0);
      }
      else
      {
        value /= 1.0 + multiplier*0.03 * ((rand()%10000)/10000.0);
      }
      date++;
    }
    date+=2;
  }
}

void read_data(std::string filename,std::vector<price> & prices)
{
  std::ifstream infile(filename.c_str());
  std::string line;
  int i=0;
  int dat=0;
  int day=0,month=0,year=0;
  while (std::getline(infile, line))
  {
    price D;
    std::stringstream iss(line);
    std::string token;
    // date
    iss >> token;
    boost::erase_all(token,"/");
    boost::erase_all(token,",");
    dat = atoi(token.c_str());
    year = dat%100;
    if(year > 50)year += 1900;else year += 2000;
    day = (dat/100)%100;
    month = (dat/10000)%100;
    //fprintf(stderr,"year:%d month:%d day:%d\n",year,month,day);
    D.date = token.c_str();
    // open
    iss >> token;
    boost::erase_all(token,",");
    D.open = atof(token.c_str());
    // high
    iss >> token;
    boost::erase_all(token,",");
    D.high = atof(token.c_str());
    // low
    iss >> token;
    boost::erase_all(token,",");
    D.low = atof(token.c_str());
    // close
    iss >> token;
    boost::erase_all(token,",");
    D.close = atof(token.c_str());
    D.volume = fabs(D.close - D.open)*100;
    prices.push_back(D);
  }
  infile.close();
  reverse(prices.begin(),prices.end());
  for(int i=0;i<prices.size();i++)
  {
    prices[i].index = i;
  }
}

void read_data_yahoo(std::string filename,std::vector<price> & prices,int synthetic_days=0)
{
  std::ifstream infile(filename.c_str());
  std::string line;
  int i=0;
  int dat=0;
  int day=0,month=0,year=0;
  while (std::getline(infile, line))
  {
    price D;
    boost::replace_all(line,","," ");
    std::stringstream iss(line);
    std::string token;
    // date
    iss >> token;
    D.date = token.c_str();
    // open
    iss >> token;
    boost::erase_all(token,",");
    D.open = atof(token.c_str());
    // high
    iss >> token;
    boost::erase_all(token,",");
    D.high = atof(token.c_str());
    // low
    iss >> token;
    boost::erase_all(token,",");
    D.low = atof(token.c_str());
    // close
    iss >> token;
    boost::erase_all(token,",");
    D.close = atof(token.c_str());
    // adj close
    iss >> token;
    boost::erase_all(token,",");
    //D.close = atof(token.c_str());
    // volume
    iss >> token;
    boost::erase_all(token,",");
    D.volume = atoi(token.c_str());
    if(D.close > 0 && D.low > 0 && D.high > 0 && D.open > 0)
    {
      prices.push_back(D);
    }
  }
  infile.close();
  for(int i=0;i<synthetic_days;i++)
  {
    price D;
    D.synthetic=true;
    D.open  =prices[prices.size()-1].open;
    D.low   =prices[prices.size()-1].low;
    D.high  =prices[prices.size()-1].high;
    D.close =prices[prices.size()-1].close;
    D.volume=prices[prices.size()-1].volume;
    prices.push_back(D);
  }
  for(int i=0;i<prices.size();i++)
  {
    prices[i].index = i;
  }
}

bool buy_only = false;
bool game_mode = false;
Scanner scanner;

std::vector<std::vector<price> > prices;
std::vector<std::string> symbols;
std::vector<int> rsymbols;
int start_date_index = 0;
int   end_date_index = 0;

struct Symbol
{
  int index;
  std::string name;
  double units;
  double buy_price;
  Symbol(double _units,double _buy_price,std::string _name,int _index)
    : units(_units)
    , buy_price(_buy_price)
    , name(_name)
    , index(_index)
  {

  }
  Symbol()
    : units(0)
    , buy_price(0)
    , name("")
    , index(-1)
  {

  }
};

struct User
{
  std::string name;
  double prev_cash;
  double cash;
  std::map<int,Symbol> rstocks;
  User(std::string _name,double _cash,std::vector<int> & _rsymbol)
  {
    name = _name;
    cash = _cash;
    for(int i=0;i<_rsymbol.size();i++)
    {
      initialize(_rsymbol[i]);
    }
  }
  void initialize(int symbol)
  {
    //rstocks.insert(std::pair<int,Symbol>(symbol,Symbol(0,0,symbols[symbol],symbol)));
    std::stringstream ss;
    ss << "Stock" << (1+rstocks.size());
    rstocks.insert(std::pair<int,Symbol>(symbol,Symbol(0,0,ss.str(),symbol)));
  }
  bool buyAll(int symbol,int date_index) // date_index counts backwards from last possible date
  {
    buy(symbol,cash/prices[symbol][prices[symbol].size()-1-date_index].high,date_index);
  }
  bool sellAll(int symbol,int date_index) // date_index counts backwards from last possible date
  {
    sell(symbol,rstocks[symbol].units,date_index);
  }
  bool buy(int symbol,double units,int date_index) // date_index counts backwards from last possible date
  {
    std::cout << prices[symbol][prices[symbol].size()-1-date_index].date << ": User <" << name << "> attempting to buy " << units << " of " << symbols[symbol] << "." << std::endl;
    std::cout << "Available cash: $" << cash << std::endl;
    std::cout << symbols[symbol] << " price: $" << prices[symbol][prices[symbol].size()-1-date_index].high << std::endl;
    std::cout << "Units: " << units << std::endl;
    std::cout << "Total price: $" << (units * prices[symbol][prices[symbol].size()-1-date_index].high) << std::endl;
    if(units * prices[symbol][prices[symbol].size()-1-date_index].high <= cash+1e5 && units > 1e-5 && cash > 1e-5)
    {
      std::cout << "Buy successful" << std::endl;
      cash -= units * prices[symbol][prices[symbol].size()-1-date_index].high;
      rstocks [symbol] . units += units;
      rstocks [symbol] . buy_price = prices[symbol][prices[symbol].size()-1-date_index].high;
      return true;
    }
    return false;
  }
  bool sell(int symbol,double units,int date_index) // date_index counts backwards from last possible date
  {
    std::cout << prices[symbol][prices[symbol].size()-1-date_index].date << ": User <" << name << "> attempting to sell " << units << " of " << symbols[symbol] << "." << std::endl;
    std::cout << symbols[symbol] << " price: $" << prices[symbol][prices[symbol].size()-1-date_index].low << std::endl;
    std::cout << "Units: " << units << std::endl;
    std::cout << "Total price: $" << (units * prices[symbol][prices[symbol].size()-1-date_index].low) << std::endl;
    if(rstocks [symbol] . units >= units-1e-5 && units > 1e-5 && rstocks[symbol] . units > 1e-5)
    {
      std::cout << "Sell successful" << std::endl;
      cash += units * prices[symbol][prices[symbol].size()-1-date_index].low;
      rstocks [symbol] . units -= units;
      rstocks [symbol] . buy_price = prices[symbol][prices[symbol].size()-1-date_index].low;
      return true;
    }
    return false;
  }
  double expected_return(int date_index)
  {
    if(cash<1e-5){}
    else{prev_cash=cash;}
    double val = cash;
    std::map<int,Symbol>::iterator it = rstocks.begin();
    while(it!=rstocks.end())
    {
      val += it->second.units * prices[it->second.index][prices[it->second.index].size()-1-date_index].low;
      ++it;
    }
    return val;
  }
};

struct Robot
{
  long num_elems;
  long num_bits;
  long num_out_bits;
  Robot()
  {
    num_elems = 0;
    num_bits = 1;
    num_out_bits = 1;
  }
  User * user;
  void predict(std::vector<double> const & input, std::vector<double> & output)
  {

  }
  void train(std::vector<double> const & input, std::vector<double> const & output)
  {

  }
  long get_input_size(long range)
  {
    std::vector<double> input;
    //for(long i=0;i<num_bits;i++)
    //    input.push_back(0/*p.CCI*/);
    //for(long i=0;i<num_bits;i++)
    //    input.push_back(0/*p.RSI*/);
    //for(long i=0;i<num_bits;i++)
    //    input.push_back(0/*p.MFI*/);
    //for(long i=0;i<num_bits;i++)
    //    input.push_back(0/*p.close > p.ema_12+1.5*p.ems_12*/);
    //for(long i=0;i<num_bits;i++)
    //    input.push_back(0/*p.MACD_dline>0&&p.MACD_dsignal>0*/);
    //for(long i=0;i<num_bits;i++)
    //    input.push_back(0/*p.MACD_dline<0&&p.MACD_dsignal<0*/);
    return input.size()+num_bits*(range);
  }
  long get_output_size(long range)
  {
    std::vector<double> output;
    return output.size()+num_bits*(range);
    //return num_out_bits*(range-1);
  }
  void encode(std::vector<double> & vec,double dat,double min_dat,double max_dat,long num)
  {
    
    if(dat<0.25*min_dat)dat=0.25*min_dat+1e-5;
    if(dat>0.25*max_dat)dat=0.25*max_dat-1e-5;
    if(dat<min_dat)dat=min_dat+1e-5;
    if(dat>max_dat)dat=max_dat-1e-5;
    double val = ((dat-min_dat)/(max_dat-min_dat));
    vec.push_back(val);
    
  }
  std::vector<double> construct_input_vector(price p,std::vector<price> const & prev)
  {
    std::vector<double> input;
    //encode(input,p.CCI,-150,150,num_bits);
    //encode(input,p.RSI,0,100,num_bits);
    //encode(input,p.MFI,0,100,num_bits);
    //encode(input,100*(p.close - p.ema_12)/p.ems_12,-150,150,num_bits);
    //encode(input,p.MACD_dline,-2,2,num_bits);
    //encode(input,p.MACD_dsignal,-2,2,num_bits);
    for(int i=0;i<prev.size();i++)
    {
      encode(input,100*prev[i].prct_change,-2,2,num_bits);
    }
    return input;
  }
  std::vector<double> construct_output_vector(price p,std::vector<price> const & next)
  {
    std::vector<double> output;
    for(int i=0;i<next.size();i++)
    {
      {
        encode(output,100*next[i].prct_change,-2,2,num_out_bits);
      }
    }
    return output;
  }
  void generate ( std::string symb
                , price p
                , std::vector<price> const & prev
                , std::vector<price> const & next
                , long & in_off
                , long & out_off
                , double * in_dump
                , double * out_dump
                , long & in_size
                , long & out_size
                , long & samples
                )
  {
    std::vector<double> input = construct_input_vector(p,prev);
    std::vector<double> output = construct_output_vector(p,next);
    in_size = input.size();
    out_size = output.size();
    {
      std::cout << symb << ":";
      std::cout << p.date << ":";
      for(int i=0;i<output.size();i++)
      {
        std::cout << ((output[i]>0.5)?"1":"0");
        out_dump[out_off+i] = ((output[i]));
      }
      std::cout << ":";
      for(int i=0;i<input.size();i++)
      {
        if(i%num_bits==0&&i>0)std::cout << "|";
        std::cout << ((input[i]>0.5)?"1":"0");
        in_dump[in_off+i] = ((input[i]));
      }
      std::cout << std::endl;
      in_off += in_size;
      out_off += out_size;
      samples ++;
      num_elems = samples;
    }
  }
};

struct mrbm_params
{

  long batch_iter;
  long num_batch;
  long total_n;
  long n;
  double epsilon;
  long n_iter;
  long n_cd;

  long v;
  long h;

  std::vector<long> input_sizes;

  std::vector<long> output_sizes;

  std::vector<long> input_iters;

  std::vector<long> output_iters;

  long bottleneck_iters;

  mrbm_params(Robot* robot,int range,long n_batch,long n_samples,double c_epsilon)
  {

    v  = robot->get_input_size(range);
    h  = robot->get_output_size(range);

    n_cd = 1;
    num_batch = n_batch;
    batch_iter = 1;
    n = n_samples;
    total_n = n_samples;
    epsilon = c_epsilon;
    n_iter = 1000;

    input_sizes.push_back(v);
    input_sizes.push_back(v);
    input_sizes.push_back(v);
    input_sizes.push_back(v);

    output_sizes.push_back(h);

    for(int i=0;i+1<input_sizes.size();i++)
        input_iters.push_back(n_iter);

    for(int i=0;i+1<output_sizes.size();i++)
        output_iters.push_back(n_iter);

    bottleneck_iters = n_iter;

  }
};

void run_mrbm(mrbm_params p,double * dat_in,double * dat_out,double * prd_out)
{
  int cd = 0;
  while(true)
  {
    if(mrbm == NULL)
    {
      mrbm = new mRBM(p.input_sizes[0],p.output_sizes[0]);
      for(long i=0;i+1<p.input_sizes.size();i++)
      {
        mrbm->input_branch.push_back(new DataUnit(p.input_sizes[i],p.input_sizes[i+1],p.input_iters[i]));
      }
      for(long i=0;i+1<p.output_sizes.size();i++)
      {
        mrbm->output_branch.push_back(new DataUnit(p.output_sizes[i],p.output_sizes[i+1],p.output_iters[i]));
      }
      long bottle_neck_num = (p.input_sizes[p.input_sizes.size()-1]+p.output_sizes[p.output_sizes.size()-1]);
      mrbm->bottle_neck = new DataUnit(p.input_sizes[p.input_sizes.size()-1]+p.output_sizes[p.output_sizes.size()-1],bottle_neck_num,p.bottleneck_iters);
    }
    mrbm->train(p.v,p.h,p.num_batch,p.total_n,(int)(p.n_cd+0.05*cd),p.epsilon/(1+0.05*cd),dat_in,dat_out);
    mrbm->model_all(p.total_n,dat_in,prd_out);
    mrbm->compare_all(p.total_n,dat_out,prd_out);
    cd ++;
  }
}

Robot * robot = new Robot();

int width  = 1000;//1800;
int height = 1000;

int mouse_x = 0;
int mouse_y = 0;

int stock_index = 0;

int start_index = -1;
int   end_index = -1;
bool pick_start_index = false;
bool   pick_end_index = false;

void drawString (void * font, char const *s, double x, double y, double z)
{
     unsigned int i;
     glRasterPos3f(x, y, z);
     for (i = 0; i < strlen (s); i++)
     {
         glutBitmapCharacter (font, s[i]);
     }
}

User * user = NULL;

long learning_samples = 0;
long test_learning_samples = 0;
int  input_learning_range = 6;
int output_learning_range = 6;
int synthetic_range = 26;
int learning_offset = 1;
int learning_num = 600;
int test_learning_num = 100;
double *  in_dump = NULL;
double * out_dump = NULL;
double *  in_test = NULL;
double * out_test = NULL;

Perceptron<double> * perceptron = NULL;
Perceptron<double> * perceptron_tmp = NULL;

void reconstruct(std::vector<price> & prices,int index)
{
    if ( perceptron != NULL 
      && in_dump 
      && out_dump 
      && prices[index].synthetic == false 
      && index+     synthetic_range*learning_offset<prices.size()+1
      && index-input_learning_range*learning_offset>=0
       )
    {
      int  in_size = robot-> get_input_size( input_learning_range);
      int out_size = robot->get_output_size(output_learning_range);
      {
        for(int j=0;j+1<synthetic_range;j++)
        {
            double * in = new double[in_size];
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            for(int i=index+1+j,k=0;k<input_learning_range;i-=learning_offset,k++)
            {
                in[k] = 100*((prices[i].synthetic)?prices[i].prct_prediction:prices[i].prct_change);
                if(in[k]<-2)in[k]=-2;
                if(in[k]> 2)in[k]= 2;
                in[k]+=2;
                in[k]/=4;
                //std::cout << "k=" << k << "\tin=" << in[k] << std::endl;
            }
            double val = 0.5;
            double tmp = 0;
            double tru = in[0];
            double min_D = tru-delta_D;
            double max_D = tru+delta_D;
            //if(prices[index+j*learning_offset].synthetic)
            {
              min_D = 0.3;
              max_D = 0.7;
            }
            {
                double err = 1e12;
                for(double D=min_D;D<=max_D;D+=0.05)
                {
                  double * tmp_in = new double[in_size];
                  for(int i=0;i<in_size;i++)
                  {
                    tmp_in[i] = in[i];
                  }
                  tmp_in[0] = D;
                  for(int iter=0;iter<100;iter++)
                  {
                    double * dat = perceptron->model(in_size,out_size,&tmp_in[0]);
                    for(int i=0;i<in_size;i++)
                    {
                      tmp_in[i] = dat[i];
                    }
                    delete [] dat;
                    dat = NULL;
                  }
                  double dat_err = 0;
                  for(int i=1;i<out_size;i++)
                  {
                    dat_err += fabs(in[i] - tmp_in[i]);
                  }
                  {
                    dat_err += fabs(D - tmp_in[0]);
                  }
                  if(dat_err<err)
                  {
                    err=dat_err;
                    val=(prices[index+1+j].synthetic)?tmp_in[0]:in[0];
                  }
                  delete [] tmp_in;
                  tmp_in = NULL;
                }
            }
            //val = in[0];
            //std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
            //std::cout << "j=" << j << "\tdat=" << dat[0] << std::endl;
            //std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
            val -= 0.5;
            val *= 4;
            val *= 0.01;
            prices[index+j*learning_offset+1]. prct_prediction = val;
            prices[index+j*learning_offset+1].close_prediction = (j==0)
                                                             ?  prices[index+j*learning_offset].close           *(1+val)
                                                             :  prices[index+j*learning_offset].close_prediction*(1+val)
                                                             ;
            delete [] in;
            in = NULL;
        }
      }
    }
}

void reconstruct_rbm(std::vector<price> & prices,int index)
{
    if ( perceptron != NULL 
      && in_dump 
      && out_dump 
      && prices[index].synthetic == false 
      && index+     synthetic_range*learning_offset<prices.size() 
      && index-input_learning_range*learning_offset>=0
       )
    {
      int  in_size = robot-> get_input_size( input_learning_range);
      int out_size = robot->get_output_size(output_learning_range);
      {
        {
            double * in = new double[in_size];
            for(int i=index,k=0;k<input_learning_range;i-=learning_offset,k++)
            {
                in[k] = 100*((prices[i].synthetic)?prices[i].prct_prediction:prices[i].prct_change);
                if(in[k]<-2)in[k]=-2;
                if(in[k]> 2)in[k]= 2;
                in[k]+=2;
                in[k]/=4;
            }
            double * dat = perceptron->model(in_size,out_size,&in[0]);
            //dat[0] -= 0.5;
            //dat[0] *= 4;
            //dat[0] *= 0.01;
            //for(int k=0;k<perceptron->n_nodes.size();k++)
            //{
            //  std::cout << k << "\t" << perceptron->n_nodes[k] << "\t" << perceptron->n_layers/2 << std::endl;
            //}
            prices[index].auto_encoding_x = 2*perceptron->activation_values1[perceptron->n_layers/2][0] - 1;
            prices[index].auto_encoding_y = 2*perceptron->activation_values1[perceptron->n_layers/2][1] - 1;
            delete [] dat;
            delete [] in;
            dat = NULL;
            in = NULL;
        }
      }
    }
}

void dump_autoencoding_to_file(std::string filename,bool quiet=false)
{
    if(!quiet)
      std::cout << "dump autoencoding to file:" << filename << std::endl;
    ofstream myfile (filename.c_str());
    if (myfile.is_open())
    {
      for(int stock=0;stock<prices.size();stock++)
      {
        for(int i=0;i<prices[stock].size();i++)
        {
          reconstruct_rbm(prices[stock],i);
          myfile << prices[stock][i].auto_encoding_x << " " << prices[stock][i].auto_encoding_y << " " << stock << " " << i << std::endl;
        }
      }
      myfile.close();
    }
    else
    {
      cout << "Unable to open file: " << filename << std::endl;
      exit(1);
    }

}

void draw_charts()
{
  glColor3f(1,1,1);
  if(!game_mode)drawString(GLUT_BITMAP_HELVETICA_18,symbols[stock_index].c_str(),-0.6,0.9,0);
  if(user&&game_mode)
  {
    {
      glColor3f(1,1,1);
      std::stringstream ss;
      ss << "Cash: $" << user->cash;
      drawString(GLUT_BITMAP_HELVETICA_18,ss.str().c_str(),-0.6,0.85,0);
    }
    std::map<int,Symbol>::iterator it = user->rstocks.begin();
    int i = 0;
    while(it != user->rstocks.end())
    {
      if(it->second.index == stock_index&&game_mode)glColor3f(0,1,0);else glColor3f(1,1,1);
      std::stringstream ss;
      ss << it->second.name << "   " << it->second.units;
      drawString(GLUT_BITMAP_HELVETICA_18,ss.str().c_str(),-0.6,0.8-i*0.05,0);
      ++it;
      ++i;
    }
    {
      glColor3f(1,1,1);
      std::stringstream ss;
      ss << "expected return: $" << user->expected_return(end_date_index);
      drawString(GLUT_BITMAP_HELVETICA_18,ss.str().c_str(),-0.6,0.8-i*0.05,0);
      ++i;
    }
    {
      glColor3f(1,1,1);
      std::stringstream ss;
      ss << "percent return: " << (100*(user->expected_return(end_date_index)/user->prev_cash) - 100) << "%";
      drawString(GLUT_BITMAP_HELVETICA_18,ss.str().c_str(),-0.6,0.8-i*0.05,0);
    }
  }
  drawString(GLUT_BITMAP_HELVETICA_18,"Volume",-1,-0.10,0);
  drawString(GLUT_BITMAP_HELVETICA_18,"MACD",-1,-0.30,0);
  drawString(GLUT_BITMAP_HELVETICA_18,"RSI",-1,-0.50,0);
  drawString(GLUT_BITMAP_HELVETICA_18,"MFI",-1,-0.70,0);
  drawString(GLUT_BITMAP_HELVETICA_18,"CCI",-1,-0.90,0);
  int n = start_date_index - end_date_index;
  if(n>=prices[stock_index].size())
  {
    n = prices[stock_index].size()-1;
  }
  int size = prices[stock_index].size()-1-end_date_index;
  double open_price = 0;
  double close_price = 0;
  double high_price = 0;
  double low_price = 0;
  int volume = 0;
  std::string date = "";
  int price_index = (int)((size-n)+(((double)mouse_x/width)*(n)));
  reconstruct(prices[stock_index],price_index);
  //reconstruct_rbm(prices[stock_index],price_index);
  if(price_index>=0&&price_index<prices[stock_index].size())
  {
    open_price  = prices[stock_index][prices[stock_index].size()-1].open;
    close_price = prices[stock_index][prices[stock_index].size()-1].close;
    high_price  = prices[stock_index][prices[stock_index].size()-1].high;
    low_price   = prices[stock_index][prices[stock_index].size()-1].low;
    volume      = prices[stock_index][prices[stock_index].size()-1].volume;
    date        = prices[stock_index][prices[stock_index].size()-1].date;
    // Current
    std::stringstream ss_date;
    ss_date << date << std::endl;
    std::stringstream ss_open_price;
    ss_open_price << "Open:" << open_price << std::endl;
    std::stringstream ss_close_price;
    ss_close_price << "Close:" << close_price << std::endl;
    std::stringstream ss_low_price;
    ss_low_price << "Low:" << low_price << std::endl;
    std::stringstream ss_high_price;
    ss_high_price << "High:" << high_price << std::endl;
    std::stringstream ss_volume;
    ss_volume << "Volume:" << volume << std::endl;
    drawString(GLUT_BITMAP_HELVETICA_18,ss_date.str().c_str()       ,-1.0f,-1.0f+0.25f,0);
    drawString(GLUT_BITMAP_HELVETICA_18,ss_open_price.str().c_str() ,-1.0f,-1.0f+0.20f,0);
    drawString(GLUT_BITMAP_HELVETICA_18,ss_close_price.str().c_str(),-1.0f,-1.0f+0.15f,0);
    drawString(GLUT_BITMAP_HELVETICA_18,ss_high_price.str().c_str() ,-1.0f,-1.0f+0.10f,0);
    drawString(GLUT_BITMAP_HELVETICA_18,ss_low_price.str().c_str()  ,-1.0f,-1.0f+0.05f,0);
    drawString(GLUT_BITMAP_HELVETICA_18,ss_volume.str().c_str()     ,-1.0f,-1.0f+0.00f,0);
  }
  if(price_index>=0&&price_index<prices[stock_index].size()&&prices[stock_index][price_index].synthetic==false)
  {
    open_price  = prices[stock_index][price_index].open;
    close_price = prices[stock_index][price_index].close;
    high_price  = prices[stock_index][price_index].high;
    low_price   = prices[stock_index][price_index].low;
    volume      = prices[stock_index][price_index].volume;
    date        = prices[stock_index][price_index].date;
    // Historical
    std::stringstream ss_date;
    ss_date << date << std::endl;
    std::stringstream ss_open_price;
    ss_open_price << "Open:" << open_price << std::endl;
    std::stringstream ss_close_price;
    ss_close_price << "Close:" << close_price << std::endl;
    std::stringstream ss_low_price;
    ss_low_price << "Low:" << low_price << std::endl;
    std::stringstream ss_high_price;
    ss_high_price << "High:" << high_price << std::endl;
    std::stringstream ss_volume;
    ss_volume << "Volume:" << volume << std::endl;
    drawString(GLUT_BITMAP_HELVETICA_18,ss_date.str().c_str()       ,-1.0f-0.15f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.25f,0);
    drawString(GLUT_BITMAP_HELVETICA_18,ss_open_price.str().c_str() ,-1.0f-0.15f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.20f,0);
    drawString(GLUT_BITMAP_HELVETICA_18,ss_close_price.str().c_str(),-1.0f-0.15f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.15f,0);
    drawString(GLUT_BITMAP_HELVETICA_18,ss_high_price.str().c_str() ,-1.0f-0.15f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.10f,0);
    drawString(GLUT_BITMAP_HELVETICA_18,ss_low_price.str().c_str()  ,-1.0f-0.15f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.05f,0);
    drawString(GLUT_BITMAP_HELVETICA_18,ss_volume.str().c_str()     ,-1.0f-0.15f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.00f,0);
  }
  VolumeByPrice vol_by_price;
  vol_by_price.create_bins(prices[stock_index],12,(int)(size-n),size);
  //double factor = 2.0f/n;
  double factor = 2.0f/(start_date_index - end_date_index + 1);
  double vfactor100 = 2.0f/100.0f;
  double vfactor400 = 2.0f/600.0f;
  double Bollinger_sigma = 1.0f;
  double vmin    = double(std::min_element(prices[stock_index].begin()+(int)(size-n),prices[stock_index].begin()+(int)(size),comparator_low )->low)/1.05f;
  double vmax    = double(std::max_element(prices[stock_index].begin()+(int)(size-n),prices[stock_index].begin()+(int)(size),comparator_high)->high)*1.05f;
  double MACD_min= double(std::min_element(prices[stock_index].begin()+(int)(size-n),prices[stock_index].begin()+(int)(size),comparator_MACD)->MACD_dline);
  double MACD_max= double(std::max_element(prices[stock_index].begin()+(int)(size-n),prices[stock_index].begin()+(int)(size),comparator_MACD)->MACD_dline);
  double MACD_cmp= max(fabs(MACD_min),fabs(MACD_max));
  double auto_min= 0;
  double auto_max= 1;
  double auto_cmp= 1;
  //std::cout << MACD_min << "\t" << MACD_max << std::endl;
  double vfactor = 2.0f/(vmax-vmin);
  double vfactor_volume = 2.0f/double(std::max_element(prices[stock_index].begin()+(int)(size-n),prices[stock_index].begin()+(int)(size),comparator_volume)->volume);

  glBegin(GL_LINES);
  // mouse 
  glColor3f(1,1,1);
  glVertex3f( 100.0f+-1.0f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-100.0f+-1.0f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-1.0f+2.0f*mouse_x/width, 100.0f+1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-1.0f+2.0f*mouse_x/width,-100.0f+1.0f-2.0f*mouse_y/height,0);
  glEnd();

  if(pick_start_index)
  {
    start_index = price_index;
    pick_start_index = false;
  }
  if(pick_end_index)
  {
    end_index = price_index;
    pick_end_index = false;
  }
  double chart_size = 0.05f;
  glBegin(GL_LINES);
  // MACD spiral axis 
  glColor3f(1.0,1.0,1.0);
  glVertex3f(-chart_size+ chart_size+-1.0f+2.0f*mouse_x/width,-chart_size            +1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+-chart_size+-1.0f+2.0f*mouse_x/width,-chart_size            +1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size            +-1.0f+2.0f*mouse_x/width,-chart_size+ chart_size+1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size            +-1.0f+2.0f*mouse_x/width,-chart_size+-chart_size+1.0f-2.0f*mouse_y/height,0);
  glEnd();
  glBegin(GL_LINES);
  // MACD current state
  glColor3f(1.0,1.0,1.0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].MACD_dline/MACD_cmp+ 0.01f+-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][price_index].MACD_dsignal)/MACD_cmp       +1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].MACD_dline/MACD_cmp+-0.01f+-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][price_index].MACD_dsignal)/MACD_cmp       +1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].MACD_dline/MACD_cmp       +-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][price_index].MACD_dsignal)/MACD_cmp+ 0.01f+1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].MACD_dline/MACD_cmp       +-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][price_index].MACD_dsignal)/MACD_cmp+-0.01f+1.0f-2.0f*mouse_y/height,0);
  glEnd();
  glBegin(GL_LINES);
  glColor3f(1.0,1.0,0.0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].auto_encoding_x/auto_cmp+ 0.01f+-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*prices[stock_index][price_index].auto_encoding_y/auto_cmp       +1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].auto_encoding_x/auto_cmp+-0.01f+-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*prices[stock_index][price_index].auto_encoding_y/auto_cmp       +1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].auto_encoding_x/auto_cmp       +-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*prices[stock_index][price_index].auto_encoding_y/auto_cmp+ 0.01f+1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].auto_encoding_x/auto_cmp       +-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*prices[stock_index][price_index].auto_encoding_y/auto_cmp+-0.01f+1.0f-2.0f*mouse_y/height,0);
  glEnd();
  glBegin(GL_LINES);
  // MACD spiral
  for(int i=1;i<n;i++)
  {
    glColor4f(1,1,1,.2f);
    if(size-i+1>start_index && size-i+1<end_index)
    {
      glColor4f(1,1,1,0.5f);
    }
    glVertex3f(-chart_size+chart_size*prices[stock_index][size-i+1].MACD_dline/MACD_cmp+-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][size-i+1].MACD_dsignal)/MACD_cmp+1.0f-2.0f*mouse_y/height,0);
    glVertex3f(-chart_size+chart_size*prices[stock_index][size-i  ].MACD_dline/MACD_cmp+-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][size-i  ].MACD_dsignal)/MACD_cmp+1.0f-2.0f*mouse_y/height,0);
  }
  glEnd();

  glBegin(GL_QUADS);
  for(int i=0;i<vol_by_price.bins.size();i++)
  {
    glColor3f(1,0,0);
    glVertex3f(-1.0f                                   , 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_min-vmin) ,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum_neg, 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_min-vmin) ,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum_neg, 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_max-vmin) ,0);
    glVertex3f(-1.0f                                   , 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_max-vmin) ,0);
    glColor3f(0,1,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum_neg, 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_min-vmin) ,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum    , 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_min-vmin) ,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum    , 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_max-vmin) ,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum_neg, 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_max-vmin) ,0);
  }
  glEnd();

  glBegin(GL_LINES);
  for(int i=1,j=0;i<n;i++,j++)
  if(!prices[stock_index][size-i+1].synthetic)
  {
    glColor3f(1,1,1);

    // volume
    glVertex3f(1.0f- j   *factor,-0.2f+0.1f*vfactor_volume*prices[stock_index][size-i+1].volume,0);
    glVertex3f(1.0f-(j+1)*factor,-0.2f+0.1f*vfactor_volume*prices[stock_index][size-i  ].volume,0);
    glVertex3f(1.0f- j   *factor,-0.2f+0.1f*vfactor_volume*prices[stock_index][size-i+1].EMAV  ,0);
    glVertex3f(1.0f-(j+1)*factor,-0.2f+0.1f*vfactor_volume*prices[stock_index][size-i  ].EMAV  ,0);

    // MACD
    glVertex3f(1.0f- j   *factor,-0.4f+0.1f*vfactor400*(300+100.0f*prices[stock_index][size-i+1].MACD_dline/MACD_cmp),0);
    glVertex3f(1.0f-(j+1)*factor,-0.4f+0.1f*vfactor400*(300+100.0f*prices[stock_index][size-i  ].MACD_dline/MACD_cmp),0);
    glColor3f(1,0,0);
    glVertex3f(1.0f- j   *factor,-0.4f+0.1f*vfactor400*(300+100.0f*prices[stock_index][size-i+1].MACD_dsignal/MACD_cmp),0);
    glVertex3f(1.0f-(j+1)*factor,-0.4f+0.1f*vfactor400*(300+100.0f*prices[stock_index][size-i  ].MACD_dsignal/MACD_cmp),0);
    glColor3f(1,1,1);
    glVertex3f(1.0f- j   *factor,-0.4f+0.1f*vfactor400*(300+0),0);
    glVertex3f(1.0f-(j+1)*factor,-0.4f+0.1f*vfactor400*(300+0),0);

    // RSI
    glVertex3f(1.0f- j   *factor,-0.6f+0.1f*vfactor100*prices[stock_index][size-i+1].RSI,0);
    glVertex3f(1.0f-(j+1)*factor,-0.6f+0.1f*vfactor100*prices[stock_index][size-i  ].RSI,0);
    glVertex3f(1.0f- j   *factor,-0.6f+0.1f*vfactor100*30,0);
    glVertex3f(1.0f-(j+1)*factor,-0.6f+0.1f*vfactor100*30,0);
    glVertex3f(1.0f- j   *factor,-0.6f+0.1f*vfactor100*50,0);
    glVertex3f(1.0f-(j+1)*factor,-0.6f+0.1f*vfactor100*50,0);
    glVertex3f(1.0f- j   *factor,-0.6f+0.1f*vfactor100*70,0);
    glVertex3f(1.0f-(j+1)*factor,-0.6f+0.1f*vfactor100*70,0);

    // MFI
    glVertex3f(1.0f- j   *factor,-0.8f+0.1f*vfactor100*prices[stock_index][size-i+1].MFI,0);
    glVertex3f(1.0f-(j+1)*factor,-0.8f+0.1f*vfactor100*prices[stock_index][size-i  ].MFI,0);
    glVertex3f(1.0f- j   *factor,-0.8f+0.1f*vfactor100*30,0);
    glVertex3f(1.0f-(j+1)*factor,-0.8f+0.1f*vfactor100*30,0);
    glVertex3f(1.0f- j   *factor,-0.8f+0.1f*vfactor100*50,0);
    glVertex3f(1.0f-(j+1)*factor,-0.8f+0.1f*vfactor100*50,0);
    glVertex3f(1.0f- j   *factor,-0.8f+0.1f*vfactor100*70,0);
    glVertex3f(1.0f-(j+1)*factor,-0.8f+0.1f*vfactor100*70,0);

    // CCI
    glVertex3f(1.0f- j   *factor,-1.0f+0.1f*vfactor400*(300+prices[stock_index][size-i+1].CCI),0);
    glVertex3f(1.0f-(j+1)*factor,-1.0f+0.1f*vfactor400*(300+prices[stock_index][size-i  ].CCI),0);
    glVertex3f(1.0f- j   *factor,-1.0f+0.1f*vfactor400*(300+ 100),0);
    glVertex3f(1.0f-(j+1)*factor,-1.0f+0.1f*vfactor400*(300+ 100),0);
    glVertex3f(1.0f- j   *factor,-1.0f+0.1f*vfactor400*(300+   0),0);
    glVertex3f(1.0f-(j+1)*factor,-1.0f+0.1f*vfactor400*(300+   0),0);
    glVertex3f(1.0f- j   *factor,-1.0f+0.1f*vfactor400*(300+-100),0);
    glVertex3f(1.0f-(j+1)*factor,-1.0f+0.1f*vfactor400*(300+-100),0);
  }
  glEnd();

  glBegin(GL_QUADS);
  for(int i=1,j=0;i<n;i++,j++)
  if(!prices[stock_index][size-i+1].synthetic)
  {
    if(prices[stock_index][size-i+1].close>prices[stock_index][size-i].close)
    {
      glColor3f(0,1,0);
    }
    else
    {
      glColor3f(1,0,0);
    }
    // volume
    glVertex3f(1.0f-(j-0.45f)*factor,-0.25f+0.125f*vfactor_volume*prices[stock_index][size-i+1].volume,0);
    glVertex3f(1.0f-(j+0.45f)*factor,-0.25f+0.125f*vfactor_volume*prices[stock_index][size-i+1].volume,0);
    glVertex3f(1.0f-(j+0.45f)*factor,-0.25f,0);
    glVertex3f(1.0f-(j-0.45f)*factor,-0.25f,0);
  }
  glEnd();

  glBegin(GL_QUADS);
  {
    glColor3f(0,0,0);
    glVertex3f(-1,-1,0);
    glVertex3f(1,-1,0);
    glVertex3f(1,0,0);
    glVertex3f(-1,0,0);
  }
  glEnd();
    
  glBegin(GL_LINES);
  for(int i=1,j=0;i<n;i++,j++)
  if(!prices[stock_index][size-i+1].synthetic)
  {
    if(prices[stock_index][size-i+1].prediction_confidence>0.5)
    {
        glColor3f(1,1,0);
    }
    else
    {
        glColor3f(.1,.1,0);
    }
    // price
    glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].close_prediction-vmin) ,0);
    glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].close_prediction-vmin) ,0);
    glColor3f(1,1,1);
    // price
    glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].close-vmin) ,0);
    glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].close-vmin) ,0);

    glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_12-vmin) ,0);
    glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_12-vmin) ,0);
    for(int b=1;b<=3;b++)
    {
      glColor3f(1.0f/(1+b),1.0f/(1+b),1.0f/(1+b));
      glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_12+b*Bollinger_sigma*prices[stock_index][size-i+1].ems_12-vmin) ,0);
      glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_12+b*Bollinger_sigma*prices[stock_index][size-i  ].ems_12-vmin) ,0);
      glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_12-b*Bollinger_sigma*prices[stock_index][size-i+1].ems_12-vmin) ,0);
      glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_12-b*Bollinger_sigma*prices[stock_index][size-i  ].ems_12-vmin) ,0);
    }
    
    //glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_26-vmin) ,0);
    //glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_26-vmin) ,0);
    //for(int b=1;b<=3;b++)
    //{
    //  glColor3f(1.0f/(1+b),1.0f/(1+b),1.0f/(1+b));
    //  glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_26+b*Bollinger_sigma*prices[stock_index][size-i+1].ems_26-vmin) ,0);
    //  glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_26+b*Bollinger_sigma*prices[stock_index][size-i  ].ems_26-vmin) ,0);
    //  glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_26-b*Bollinger_sigma*prices[stock_index][size-i+1].ems_26-vmin) ,0);
    //  glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_26-b*Bollinger_sigma*prices[stock_index][size-i  ].ems_26-vmin) ,0);
    //}
    
    //glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_50-vmin) ,0);
    //glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_50-vmin) ,0);
    //for(int b=1;b<=3;b++)
    //{
    //  glColor3f(1.0f/(1+b),1.0f/(1+b),1.0f/(1+b));
    //  glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_50+b*Bollinger_sigma*prices[stock_index][size-i+1].ems_50-vmin) ,0);
    //  glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_50+b*Bollinger_sigma*prices[stock_index][size-i  ].ems_50-vmin) ,0);
    //  glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_50-b*Bollinger_sigma*prices[stock_index][size-i+1].ems_50-vmin) ,0);
    //  glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_50-b*Bollinger_sigma*prices[stock_index][size-i  ].ems_50-vmin) ,0);
    //}
  }
  else
  {
    if(prices[stock_index][size-i+1].prediction_confidence>0.5)
    {
        glColor3f(1,1,0);
    }
    else
    {
        glColor3f(.1,.1,0);
    }
    // price
    glVertex3f(1.0f- j   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].close_prediction-vmin) ,0);
    glVertex3f(1.0f-(j+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].close_prediction-vmin) ,0);
  } 
  glEnd();

  glBegin(GL_QUADS);
  for(int i=1,j=0;i<n;i++,j++)
  if(!prices[stock_index][size-i+1].synthetic)
  {
    if(size-i+1 == price_index)
    {
      if(prices[stock_index][size-i+1].close>prices[stock_index][size-i+1].open)
      {
        glColor3f(0,1,0);
      }
      else
      {
        glColor3f(1,0,0);
      }
    }
    else
    {
      if(prices[stock_index][size-i+1].close>prices[stock_index][size-i+1].open)
      {
        glColor3f(0,1,1);
      }
      else
      {
        glColor3f(1,0.5f,0);
      }
    }
    // price
    glVertex3f(1.0f-(j-0.45f)*factor, 0.0f+0.5f*vfactor*(prices[stock_index][size-i+1].open -vmin) ,0);
    glVertex3f(1.0f-(j+0.45f)*factor, 0.0f+0.5f*vfactor*(prices[stock_index][size-i+1].open -vmin) ,0);
    glVertex3f(1.0f-(j+0.45f)*factor, 0.0f+0.5f*vfactor*(prices[stock_index][size-i+1].close-vmin) ,0);
    glVertex3f(1.0f-(j-0.45f)*factor, 0.0f+0.5f*vfactor*(prices[stock_index][size-i+1].close-vmin) ,0);

    glVertex3f(1.0f-(j-0.05f)*factor, 0.0f+0.5f*vfactor*(prices[stock_index][size-i+1].low  -vmin) ,0);
    glVertex3f(1.0f-(j+0.05f)*factor, 0.0f+0.5f*vfactor*(prices[stock_index][size-i+1].low  -vmin) ,0);
    glVertex3f(1.0f-(j+0.05f)*factor, 0.0f+0.5f*vfactor*(prices[stock_index][size-i+1].high -vmin) ,0);
    glVertex3f(1.0f-(j-0.05f)*factor, 0.0f+0.5f*vfactor*(prices[stock_index][size-i+1].high -vmin) ,0);
  }
  glEnd();

}

long construct_learning_data(int start,int num,int offset,double * in_dump,double * out_dump)
{
  std::cout << "construct learning data" << std::endl;
  long in_off = 0;
  long out_off = 0;
  long in_size = 0;
  long out_size = 0;
  long samples = 0;
  for(int stock_index=0;stock_index<prices.size();stock_index++)
  {
    if(prices[stock_index].size()>num-offset*synthetic_range)
    {
      long size = prices[stock_index].size()-1;
      for(long ind=(start+input_learning_range)*offset;ind<num;ind++)
      {
        bool go = true;
        std::vector<price> prev_prcs;
        for(long k=0;k<input_learning_range;k++)
        {
          if(prices[stock_index][size-offset*k-ind].synthetic){go=false;break;}
          prev_prcs.push_back(prices[stock_index][size-offset*k-ind]);
        }
        if(go==false)continue;
        std::vector<price> next_prcs;
        for(long k=1;k<output_learning_range;k++)
        {
          if(prices[stock_index][size+offset*k-ind].synthetic){go=false;break;}
          next_prcs.push_back(prices[stock_index][size+offset*k-ind]);
        }
        if(go==false)continue;
        robot->generate ( symbols[stock_index]
                        , prices[stock_index][size-ind]
                        , prev_prcs
                        , prev_prcs // next_prcs
                        , in_off
                        , out_off
                        , in_dump
                        , out_dump
                        , in_size
                        , out_size
                        , samples
                        );
      }
    }
  }
  std::cout << in_off << "\t" << in_size << "\t" << robot->get_input_size(input_learning_range) << std::endl;
  std::cout << out_off << "\t" << out_size << "\t" << robot->get_output_size(output_learning_range) << std::endl;
  std::cout << "done constructing learning data" << std::endl;
  return samples;
}

long learning_selection = 0;

double * err_stats = NULL;
bool err_stats_changed = false;
long err_stats_cnt = 1;

double * min_variable = NULL;
double * max_variable = NULL;

int x_dim = 0;
int y_dim = 1;

void init_energy()
{

  if(perceptron != NULL)
  {
    min_variable = new double[perceptron->get_num_variables()];
    max_variable = new double[perceptron->get_num_variables()];
    for(int i=0;i<perceptron->get_num_variables();i++)
    {
        min_variable[i] = -20;
        max_variable[i] =  20;
    }
  }

}

void draw_energy()
{

  if(  perceptron != NULL 
    && x_dim >= 0 
    && x_dim < perceptron->get_num_variables() 
    && y_dim >= 0 
    && y_dim < perceptron->get_num_variables() 
    )
  {
    static bool init_eng = true;
    if(init_eng)
    {
        init_eng = false;
        init_energy();
    }
    glColor3f(1,1,1);
    std::stringstream ss0,ss1,ss2,ss3;
    ss0 << ((int)(100*min_variable[x_dim])/100.0f);
    if(!game_mode)drawString(GLUT_BITMAP_HELVETICA_18,ss0.str().c_str(),-0.9,-0.8,0);
    ss1 << ((int)(100*max_variable[x_dim])/100.0f);
    if(!game_mode)drawString(GLUT_BITMAP_HELVETICA_18,ss1.str().c_str(),-0.9, 0.8,0);
    ss2 << ((int)(100*min_variable[y_dim])/100.0f);
    if(!game_mode)drawString(GLUT_BITMAP_HELVETICA_18,ss2.str().c_str(),-0.8,-0.9,0);
    ss3 << ((int)(100*max_variable[y_dim])/100.0f);
    if(!game_mode)drawString(GLUT_BITMAP_HELVETICA_18,ss3.str().c_str(), 0.8,-0.9,0);
    glColor3f(1-perceptron->final_error,1-perceptron->final_error,1-perceptron->final_error);
    glBegin(GL_QUADS);
    glVertex3f(
        -1 + 2*(perceptron->get_variable(x_dim)-min_variable[x_dim])/(max_variable[x_dim]-min_variable[x_dim]) - 0.005 ,
        -1 + 2*(perceptron->get_variable(y_dim)-min_variable[y_dim])/(max_variable[y_dim]-min_variable[y_dim]) - 0.005 ,
        0
    );
    glVertex3f(
        -1 + 2*(perceptron->get_variable(x_dim)-min_variable[x_dim])/(max_variable[x_dim]-min_variable[x_dim]) - 0.005 ,
        -1 + 2*(perceptron->get_variable(y_dim)-min_variable[y_dim])/(max_variable[y_dim]-min_variable[y_dim]) + 0.005 ,
        0
    );
    glVertex3f(
        -1 + 2*(perceptron->get_variable(x_dim)-min_variable[x_dim])/(max_variable[x_dim]-min_variable[x_dim]) + 0.005 ,
        -1 + 2*(perceptron->get_variable(y_dim)-min_variable[y_dim])/(max_variable[y_dim]-min_variable[y_dim]) + 0.005 ,
        0
    );
    glVertex3f(
        -1 + 2*(perceptron->get_variable(x_dim)-min_variable[x_dim])/(max_variable[x_dim]-min_variable[x_dim]) + 0.005 ,
        -1 + 2*(perceptron->get_variable(y_dim)-min_variable[y_dim])/(max_variable[y_dim]-min_variable[y_dim]) - 0.005 ,
        0
    );
    glEnd();
    
    // list all files in current directory.
    boost::filesystem::path p ("snapshots");
    boost::filesystem::directory_iterator end_itr;
    // cycle through the directory
    for (boost::filesystem::directory_iterator itr(p); itr != end_itr; ++itr)
    {
      // If it's not a directory, list it. If you want to list directories too, just remove this check.
      if (boost::filesystem::is_regular_file(itr->path())) {
        std::string current_file = itr->path().string();
        perceptron_tmp -> load_from_file(current_file,true);
        glColor3f(1-perceptron_tmp->final_error,1-perceptron_tmp->final_error,1-perceptron_tmp->final_error);
        glBegin(GL_QUADS);
        glVertex3f(
            -1 + 2*(perceptron_tmp->get_variable(x_dim)-min_variable[x_dim])/(max_variable[x_dim]-min_variable[x_dim]) - 0.005 ,
            -1 + 2*(perceptron_tmp->get_variable(y_dim)-min_variable[y_dim])/(max_variable[y_dim]-min_variable[y_dim]) - 0.005 ,
            0
        );
        glVertex3f(
            -1 + 2*(perceptron_tmp->get_variable(x_dim)-min_variable[x_dim])/(max_variable[x_dim]-min_variable[x_dim]) - 0.005 ,
            -1 + 2*(perceptron_tmp->get_variable(y_dim)-min_variable[y_dim])/(max_variable[y_dim]-min_variable[y_dim]) + 0.005 ,
            0
        );
        glVertex3f(
            -1 + 2*(perceptron_tmp->get_variable(x_dim)-min_variable[x_dim])/(max_variable[x_dim]-min_variable[x_dim]) + 0.005 ,
            -1 + 2*(perceptron_tmp->get_variable(y_dim)-min_variable[y_dim])/(max_variable[y_dim]-min_variable[y_dim]) + 0.005 ,
            0
        );
        glVertex3f(
            -1 + 2*(perceptron_tmp->get_variable(x_dim)-min_variable[x_dim])/(max_variable[x_dim]-min_variable[x_dim]) + 0.005 ,
            -1 + 2*(perceptron_tmp->get_variable(y_dim)-min_variable[y_dim])/(max_variable[y_dim]-min_variable[y_dim]) - 0.005 ,
            0
        );
        glEnd();
      }
    }

    usleep(1000);

  }

}

double min_D=0;
double max_D=1;
int selection_mode = 0;

void evaluate_prediction()
{
    if(perceptron != NULL)
    {
        std::cout << "Evaluating prediction confidence" << std::endl;
        int  in_size = robot-> get_input_size( input_learning_range);
        int out_size = robot->get_output_size(output_learning_range);
        for(int stock=0;stock<prices.size();stock++)
        {
            double err_prct = 0;
            for(int index=0;index<prices[stock].size();index++)
            {
                //std::cout << symbols[stock] << '\t' << index << std::endl;
                double * in = new double[in_size];
                bool go = true;
                for(int i=index+1,k=0;k<input_learning_range;i-=learning_offset,k++)
                {
                    if(i<0||i>=prices[stock].size())
                    {
                        go = false;
                        break;
                    }
                    in[k] = 100*((prices[stock][i].synthetic)?prices[stock][i].prct_prediction:prices[stock][i].prct_change);
                    if(in[k]<-2)in[k]=-2;
                    if(in[k]> 2)in[k]= 2;
                    in[k]+=2;
                    in[k]/=4;
                }
                if(!go)continue;
                double dat_min_err = 1e10;
                double del2 = 0.05;
                for(double D=0.0;D<=1.0;D+=del2)
                {
                  double * tmp_in = new double[in_size];
                  for(int i=0;i<in_size;i++)
                  {
                    tmp_in[i] = in[i];
                  }
                  tmp_in[0] = D;
                  for(int iter=0;iter<100;iter++)
                  {
                    double * dat = perceptron->model2(in_size,out_size,&tmp_in[0]);
                    for(int i=0;i<in_size;i++)
                    {
                      tmp_in[i] = dat[i];
                    }
                    delete [] dat;
                    dat = NULL;
                  }
                  double dat_err = 0;
                  for(int j=1;j<out_size;j++)
                  {
                    dat_err += fabs(in[j] - tmp_in[j]);
                  }
                  {
                    dat_err += fabs(D - tmp_in[0]);
                  }
                  {
                    if(dat_err < dat_min_err)
                    {
                      dat_min_err = dat_err;
                      prices[stock][index+1].prediction_confidence = (fabs(D - in[0])<0.11)?1:0;
                    }
                  }
                  delete [] tmp_in;
                  tmp_in = NULL;
                }
                delete [] in;
                in = NULL;
                err_prct += prices[stock][index+1].prediction_confidence;
            }
            std::cout << "Rate:" << err_prct / prices[stock].size() << std::endl;
        }
        std::cout << "Done" << std::endl;
    }
}

void draw_learning_progress()
{
  if(in_dump&&out_dump)
  {
    int  in_size = robot-> get_input_size( input_learning_range);
    int out_size = robot->get_output_size(output_learning_range);
    // draw input vector
    {
        double dx=0.5f/in_size;
        double val;
        glBegin(GL_QUADS);
        for(int x=0;x<in_size;x++)
        {
            val = in_dump[learning_selection*in_size+x];
            glColor3f(val,val,val);
            glVertex3f(-1+0.2+ x   *dx,-1+.2     ,0);
            glVertex3f(-1+0.2+(x+1)*dx,-1+.2     ,0);
            glVertex3f(-1+0.2+(x+1)*dx,-1+.2+0.01,0);
            glVertex3f(-1+0.2+ x   *dx,-1+.2+0.01,0);
        }
        glEnd();
    }

    // draw output vector
    {
        double dx=0.5f/out_size;
        glBegin(GL_QUADS);
        for(int x=0;x<out_size;x++)
        {
            double val = out_dump[learning_selection*out_size+x];
            glColor3f(val,val,val);
            glVertex3f(-1+1+0.2+ x   *dx,-1+.2     ,0);
            glVertex3f(-1+1+0.2+(x+1)*dx,-1+.2     ,0);
            glVertex3f(-1+1+0.2+(x+1)*dx,-1+.2+0.01,0);
            glVertex3f(-1+1+0.2+ x   *dx,-1+.2+0.01,0);
        }
        glEnd();
    }

    // draw learning states
    if(mrbm != NULL)
    {
      if(mrbm->model_ready == true)
      {
        double dx=0.5f/in_size;
        //std::cout << "in_dump:\t";
        //for(int i=0;i<in_size;i++)
        //std::cout << in_dump[learning_selection*in_size+i] << '\t';
        //std::cout << '\n';
        double ** dat = mrbm->model(learning_selection,in_dump);
        //std::cout << "dat:\t\t";
        //for(int i=0;i<in_size;i++)
        //std::cout << dat[0][i] << '\t';
        //std::cout << '\n';
        for(int l=0;l<20;l++)
        {
          glBegin(GL_QUADS);
          for(int x=0;x<in_size+out_size;x++)
          {
              double val = 0.2+0.8*dat[l][x];
              if(val<0)val=0;
              if(val>1)val=1;
              {
                  glColor3f(val,val,val);
              }
              glVertex3f(-1+0.2+ x   *dx,-1+.2+0.02*(l+1)     ,0);
              glVertex3f(-1+0.2+(x+1)*dx,-1+.2+0.02*(l+1)     ,0);
              glVertex3f(-1+0.2+(x+1)*dx,-1+.2+0.02*(l+1)+0.01,0);
              glVertex3f(-1+0.2+ x   *dx,-1+.2+0.02*(l+1)+0.01,0);
          }
          glEnd();
        }
        for(int i=0;i<20;i++)delete [] dat[i];
        delete [] dat;
        dat = NULL;
      }
    }

    if(perceptron != NULL)
    {
      {
        //if(err_stats == NULL)
        //{
        //  err_stats = new double[out_size];
        //  for(int x=0;x<out_size;x++)
        //  {
        //    err_stats[x] = 0.5; 
        //  }
        //}
        double dx=0.5f/out_size;
        double * in = new double[in_size];
        double * dat_fin = new double[in_size];
        for(int i=0;i<in_size;i++)
        {
          in[i] = in_dump[in_size*learning_selection+i];
        }
        double tru_err=100;
        double dat_err=0;
        int it = 0;
        glBegin(GL_LINES);
        double tru = in[0];
        double T = 0;
        double R = 0;
        {
            //double del = 0.01;
            //int n_distribution = (int)(1/del);
            //double * distribution = new double[n_distribution];
            //for(int i=0;i<n_distribution;i++)
            //{
            //    distribution[i] = 1;
            //}
            //double count = 0;
            //for(int i=0;i<n_distribution;i++)
            //{
            //    count += distribution[i];
            //}
            //for(int iter=0;iter<10000;iter++)
            //{
            //    //std::cout << "iter=" << iter << std::endl;
            //    double ret = 0;
            //    int it = 0;
            //    for(;it<n_distribution;it++)
            //    {
            //        double D = 0.3+0.4*((rand()%10000)/10000.0);
            //        in[0] = D;
            //        double * dat = perceptron->model(in_size,out_size,&in[0]);
            //        ret += distribution[it] * dat[0] / count;
            //        int I = dat[0]*n_distribution;
            //        if(I>=0&&I<n_distribution)
            //        distribution[I]++;
            //        count++;
            //        delete [] dat;
            //        dat = NULL;
            //    }
            //    double max_distribution = 0;
            //    for(int i=0;i<n_distribution;i++)
            //    {
            //        if(distribution[i]>max_distribution)
            //        {
            //            max_distribution = distribution[i];
            //            R = i*del;
            //        }
            //    }
            //}
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //for(int i=0;i<n_distribution;i++)
            //{
            //    std::cout << i << "\t" << distribution[i] << std::endl;
            //}
            ////char ch;
            ////std::cin >> ch;
            //delete [] distribution;
            double prev_dat_err=1;
            double dat_min_err = 1e10;
            double del2 = 0.01;
            for(double D=0.0;D<=1.0;D+=del2)
            {
              double * tmp_in = new double[in_size];
              for(int i=0;i<in_size;i++)
              {
                tmp_in[i] = in[i];
              }
              tmp_in[0] = D;
              for(int iter=0;iter<1000;iter++)
              {
                double * dat = perceptron->model(in_size,out_size,&tmp_in[0]);
                for(int i=0;i<in_size;i++)
                {
                  tmp_in[i] = dat[i];
                }
                delete [] dat;
                dat = NULL;
              }
              dat_err = 0;
              for(int j=1;j<out_size;j++)
              {
                dat_err += fabs(in[j] - tmp_in[j]);
              }
              {
                dat_err += fabs(D - tmp_in[0]);
              }
              glColor3f(1,1,1);
              glVertex3f(-1+2*(D-del2),-1+2*prev_dat_err,0);
              glVertex3f(-1+2*(D),-1+2*dat_err,0);
              prev_dat_err=dat_err;
              {
                if(dat_err < dat_min_err)
                {
                  T = D;
                  dat_min_err = dat_err;
                  for(int j=0;j<out_size;j++)
                  {
                    dat_fin[j] = tmp_in[j];
                  }
                }
              }
              delete [] tmp_in;
              tmp_in = NULL;
            }
        }
        {
              double * tmp_in = new double[in_size];
              for(int i=0;i<in_size;i++)
              {
                tmp_in[i] = in[i];
              }
              tmp_in[0] = 0.5;
              for(int iter=0;iter<1000;iter++)
              {
                double * dat = perceptron->model(in_size,out_size,&tmp_in[0]);
                for(int i=0;i<in_size;i++)
                {
                  tmp_in[i] = dat[i];
                }
                delete [] dat;
                dat = NULL;
              }
              delete [] tmp_in;
              tmp_in = NULL;
        }
        glColor3f(1,1,0);
        glVertex3f(-1+2*tru,-1,0);
        glVertex3f(-1+2*tru,1,0);
        glColor3f(0,1,0);
        glVertex3f(-1+2*T,-1,0);
        glVertex3f(-1+2*T,1,0);
        glColor3f(0,0,1);
        glVertex3f(-1+2*R,-1,0);
        glVertex3f(-1+2*R,1,0);
        glEnd();
        if(fabs(T-tru) > 0.05)
        {
            learning_selection++;if(learning_selection>=learning_samples)learning_selection=0;
        }
        {
          glBegin(GL_QUADS);
          for(int x=0;x<out_size;x++)
          {
              double val = dat_fin[x];
              glColor3f(val,val,val);
              glVertex3f(-1+1+0.2+ x   *dx,-1+.2+0.02     ,0);
              glVertex3f(-1+1+0.2+(x+1)*dx,-1+.2+0.02     ,0);
              glVertex3f(-1+1+0.2+(x+1)*dx,-1+.2+0.02+0.01,0);
              glVertex3f(-1+1+0.2+ x   *dx,-1+.2+0.02+0.01,0);
          }
          glEnd();
        }
        delete [] dat_fin;
        delete [] in;
        dat_fin = NULL;
        in = NULL;

        for(int layer=0;layer<perceptron->n_nodes.size();layer++)
        {
            double dx=0.5f/perceptron->n_nodes[layer];
            double val;
            glBegin(GL_QUADS);
            for(int x=0;x<perceptron->n_nodes[layer];x++)
            {
                val = perceptron->activation_values1[layer][x];
                glColor3f(val,val,val);
                glVertex3f(-1+0.2+ x   *dx,-1+.2+(layer+1)*0.02     ,0);
                glVertex3f(-1+0.2+(x+1)*dx,-1+.2+(layer+1)*0.02     ,0);
                glVertex3f(-1+0.2+(x+1)*dx,-1+.2+(layer+1)*0.02+0.01,0);
                glVertex3f(-1+0.2+ x   *dx,-1+.2+(layer+1)*0.02+0.01,0);
            }
            glEnd();
        }

      }
    }

    // draw errors
    {
      double max_err = 0;
      for(long k=0;k<errs.size();k++)
      {
        if(max_err<errs[k])max_err=errs[k];
      }
      glBegin(GL_LINES);
      for(long k=0;k+1<errs.size();k++)
      {
        glColor3f(1,1,1);
        glVertex3f( -1 + 2*k / ((double)errs.size()-1)
                  , errs[k] / max_err
                  , 0
                  );
        glVertex3f( -1 + 2*(k+1) / ((double)errs.size()-1)
                  , errs[k+1] / max_err
                  , 0
                  );
        glVertex3f( -1 + 2*k / ((double)errs.size()-1)
                  , 0
                  , 0
                  );
        glVertex3f( -1 + 2*(k+1) / ((double)errs.size()-1)
                  , 0
                  , 0
                  );
        glColor3f(1,1,0);
        glVertex3f( -1 + 2*k / ((double)errs.size()-1)
                  , test_errs[k] / max_err
                  , 0
                  );
        glVertex3f( -1 + 2*(k+1) / ((double)errs.size()-1)
                  , test_errs[k+1] / max_err
                  , 0
                  );
        glVertex3f( -1 + 2*k / ((double)errs.size()-1)
                  , 0
                  , 0
                  );
        glVertex3f( -1 + 2*(k+1) / ((double)errs.size()-1)
                  , 0
                  , 0
                  );
      }
      glEnd();
    }

  }
}

int draw_mode = 1;

void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if(draw_mode == 0)
  {
    draw_charts();
  }
  else
  if(draw_mode == 1)
  {
    draw_learning_progress();
  }
  else
  if(draw_mode == 2)
  {
    draw_energy();
  }
  glutSwapBuffers();
}

void idle()
{
  usleep(10000);
  glutPostRedisplay();
}

void init()
{
  /* Use depth buffering for hidden surface elimination. */
  glEnable(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 10.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 0.0, 1.8,  /* eye is at (0,0,5) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 1.0, 0.);      /* up is in positive Y direction */

  /* Adjust cube position to be asthetic angle. */
  glTranslatef(0.0, 0.0, -1.0);
  glRotatef(0, 1.0, 0.0, 0.0);
  glRotatef(0, 0.0, 0.0, 1.0);
  glEnable (GL_BLEND); 
  //glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE);
  glBlendEquation(GL_FUNC_ADD);
}

std::string input_filename = "";

void run_perceptron()
{
    std::cout << "learning samples: " << learning_samples << "\t" << test_learning_samples << std::endl;
    int  num_inputs = robot-> get_input_size(input_learning_range);
    int num_outputs = robot->get_output_size(output_learning_range);
    std::vector<int> layer;
    int N = 21;
    layer.push_back(N);
    layer.push_back(N);
    //layer.push_back(N);
    //layer.push_back(N);
    //while(N>2)
    //{
    //  N -= 2;
    //  if(N>=2)
    //  {
    //    layer.push_back(N);
    //  }
    //}
    std::vector<int> num_hidden;
    for(int i=0;i<layer.size();i++)
    {
      num_hidden.push_back(layer[i]);
    }
    for(int i=0;i+1<layer.size();i++)
    {
      num_hidden.push_back(layer[(int)layer.size()-2-i]);
    }
    //for(int i=0;i<num_hidden.size();i++)
    //{
    //  std::cout << i << "\t" << num_hidden[i] << std::endl;
    //}
    long num_ann_iters = 100000;
    std::vector<long> nodes;
    nodes.push_back(num_inputs); // inputs
    for(int h=0;h<num_hidden.size();h++)
      nodes.push_back(num_hidden[h]); // hidden layer
    nodes.push_back(num_outputs); // output layer
    nodes.push_back(num_outputs); // outputs

    std::vector<RBM*> rbms;
    double ** out = new double*[num_hidden.size()/2+1];
    for(int r=0;r<num_hidden.size()/2+1;r++)
    {
        RBM * rbm = new RBM((r==0)?num_inputs:num_hidden[r-1],num_hidden[r],learning_samples,(r==0)?in_dump:out[r-1]);
        int n_iters = 20000;
        for(int i=0;i<n_iters;i++)
        {
          if(i%100==0)std::cout << i << "\t" << n_iters << std::endl;
          rbm->init(0);
          rbm->cd(1+(int)(i/6000),0.1,0);
        }
        rbms.push_back(rbm);
        out[r] = new double[num_hidden[r]*learning_samples];
        rbm->vis2hid((r==0)?in_dump:out[r-1],out[r]);
    }

    int n_prptn = 1;//000;
    perceptron_tmp = new Perceptron<double>(nodes);
    Perceptron<double> ** p = new Perceptron<double>*[n_prptn];
    double min_err = 1e10;
    double tmp_err;
    int min_ind = 0;
    for(int i=0;i<n_prptn;i++)
    {
        p[i] = new Perceptron<double>(nodes);
        p[i]->epsilon = 0.01;
        p[i]->alpha = 0.01;
        p[i]->sigmoid_type = 0;
        if(input_filename.size()>0)
        {
          p[i]->load_from_file(input_filename);
        }
        perceptron = p[i];
        {
            int layer = 0;
            for(int r=0;r<rbms.size();r++)
            {
              {
                for(int i=0;i<nodes[layer+1];i++)
                {
                  for(int j=0;j<nodes[layer];j++)
                  {
                    perceptron->weights_neuron[layer][i][j] = rbms[r]->W[j*rbms[r]->h+i];
                  }
                  perceptron->weights_bias[layer][i] = rbms[r]->c[i];
                }
              }
              layer++;
            }
            for(int r=rbms.size()-1;r>=0;r--)
            {
              {
                for(int i=0;i<nodes[layer+1];i++)
                {
                  for(int j=0;j<nodes[layer];j++)
                  {
                    perceptron->weights_neuron[layer][i][j] = rbms[r]->W[i*rbms[r]->h+j];
                  }
                  perceptron->weights_bias[layer][i] = rbms[r]->b[i];
                }
              }
              layer++;
            }
        }
        //p[i]->train(p[i]->sigmoid_type,p[i]->epsilon,1000,learning_samples,test_learning_samples,num_inputs,in_dump,in_test,num_outputs,out_dump,out_test,(i==0)?NULL:p[0]->quasi_newton);
        if(p[i]->final_error<min_err)
        {
            min_err = p[i]->final_error;
            min_ind = i;
        }
    }
    perceptron = p[min_ind];
    //perceptron->train(perceptron->sigmoid_type,perceptron->epsilon,num_ann_iters,learning_samples,test_learning_samples,num_inputs,in_dump,in_test,num_outputs,out_dump,out_test,p[0]->quasi_newton);
    evaluate_prediction();
}

bool training = false;
void keyboard(unsigned char key,int x,int y)
{
  switch(key)
  {
    case 't':delta_D *= 1.1;break;
    case 'h':delta_D /= 1.1;break;
    case 'n':selection_mode=(selection_mode+1)%3;
             switch(selection_mode)
             {
                case 0:min_D=0.3;max_D=0.43;break;
                case 1:min_D=0.43;max_D=0.56;break;
                case 2:min_D=0.56;max_D=0.7;break;
             }
             break;
    case 'x':dump_autoencoding_to_file("autoencoding.csv");break;
    case '5':continue_training=!continue_training;std::cout << "continue training:" << continue_training << std::endl;break;
    case '6':stop_training=!stop_training;std::cout << "stop training:" << stop_training << std::endl;break;
    case '-':x_dim--;std::cout << x_dim << "\t" << y_dim << '\t' << perceptron->get_num_variables() << std::endl;break;
    case '=':x_dim++;std::cout << x_dim << "\t" << y_dim << '\t' << perceptron->get_num_variables() << std::endl;break;
    case '[':y_dim--;std::cout << x_dim << "\t" << y_dim << '\t' << perceptron->get_num_variables() << std::endl;break;
    case ']':y_dim++;std::cout << x_dim << "\t" << y_dim << '\t' << perceptron->get_num_variables() << std::endl;break;
    case '9':perceptron->dump_to_file("network.ann.new");break;
    case '0':perceptron->load_from_file("network.ann");break;
    case '\'':if(perceptron->quasi_newton!=NULL){perceptron->quasi_newton->quasi_newton_update=!perceptron->quasi_newton->quasi_newton_update;}break;
    case ';':perceptron->sigmoid_type = (perceptron->sigmoid_type+1)%3;break;
    case '3':perceptron->epsilon /= 1.1;break;
    case '4':perceptron->epsilon *= 1.1;break;
    case '7':damp_weight /= 1.1;if(damp_weight>1)damp_weight=1;break;
    case '8':damp_weight *= 1.1;if(damp_weight>1)damp_weight=1;break;
    //case '1':sample_index--;if(sample_index<0)sample_index=0;break;
    //case '2':sample_index++;if(sample_index>=robot->get_output_size(output_learning_range))sample_index=robot->get_output_size(output_learning_range)-1;break;
    case '1':train_index--;if(train_index<0)train_index=0;break;
    case '2':train_index++;if(train_index>=robot->get_output_size(output_learning_range))train_index=robot->get_output_size(output_learning_range)-1;break;
    case 'r':perceptron->alpha *= 1.1;perceptron->quasi_newton->alpha = perceptron->alpha; break;
    case 'f':perceptron->alpha /= 1.1;perceptron->quasi_newton->alpha = perceptron->alpha; break;
    case 'p':
      {
        if(training==false)
        {
          training=true;
          /*
          mrbm_params params(robot,learning_range,learning_samples-1,learning_samples,.5);
          boost::thread * thr ( new boost::thread ( run_mrbm
                                                  , params
                                                  , in_dump
                                                  , out_dump
                                                  , prd_dump
                                                  ) 
                              );
          */
          boost::thread * thr ( new boost::thread ( run_perceptron ) );
        }
        break;
      }
      break;
    case 'k':learning_selection++;if(learning_selection>=learning_samples)learning_selection=learning_samples-1;else err_stats_changed=true;break;
    case 'j':learning_selection--;if(learning_selection<0)learning_selection=0;else err_stats_changed=true;break;
    case 'm': // chage draw mode
      draw_mode = (draw_mode+1)%3;
      break;
    case 'y': // buy
      if(game_mode&&user){
        user->buyAll(stock_index,end_date_index);
      }
      break;
    case 'u': // sell
      if(game_mode&&user){
        user->sellAll(stock_index,end_date_index);
      }
      break;
    case ' ':
      if(start_date_index>0&&end_date_index>0)
      {
        start_date_index--;
          end_date_index--;
      }
      break;
    case 'a':
      if(buy_only==false&&game_mode==false)
      {
        stock_index--;
        if(stock_index<0)stock_index=prices.size()-1;
      }
      if(buy_only)
      {
        if(scanner.buy.size()==0)
        {
          buy_only=false;
          stock_index--;
          if(stock_index<0)stock_index=prices.size()-1;
        }
        else
        {
          while(true)
          {
            stock_index--;
            if(stock_index<0)stock_index=prices.size()-1;
            if(scanner.buy.find(symbols[stock_index]) != scanner.buy.end())break;
          }
        }
      }
      if(game_mode)
      {
        if(rsymbols.size()==0)
        {
          game_mode=false;
          stock_index--;
          if(stock_index<0)stock_index=prices.size()-1;
        }
        else
        {
          while(true)
          {
            stock_index--;
            if(stock_index<0)stock_index=prices.size()-1;
            if(std::find(rsymbols.begin(),rsymbols.end(),stock_index) != rsymbols.end())break;
          }
        }
      }
      std::cout << symbols[stock_index] << std::endl;
      break;
    case 'd':
      if(buy_only==false&&game_mode==false)
      {
        stock_index++;
        if(stock_index>=prices.size())stock_index=0;
      }
      if(buy_only)
      {
        if(scanner.buy.size()==0)
        {
          buy_only=false;
          stock_index++;
          if(stock_index>=prices.size())stock_index=0;
        }
        else
        {
          while(true)
          {
            stock_index++;
            if(stock_index>=prices.size())stock_index=0;
            if(scanner.buy.find(symbols[stock_index]) != scanner.buy.end())break;
          }
        }
      }
      if(game_mode)
      {
        if(rsymbols.size()==0)
        {
          game_mode=false;
          stock_index++;
          if(stock_index>=prices.size())stock_index=0;
        }
        else
        {
          while(true)
          {
            stock_index++;
            if(stock_index>=prices.size())stock_index=0;
            if(std::find(rsymbols.begin(),rsymbols.end(),stock_index) != rsymbols.end())break;
          }
        }
      }
      std::cout << symbols[stock_index] << std::endl;
      break;
    case 'g':
      game_mode = !game_mode;
      if(game_mode) 
      {
        start_date_index = 3000 - (rand()%1000);
        end_date_index = start_date_index - 100;
        if(rsymbols.size()==0)
        {
          game_mode=false;
          stock_index++;
          if(stock_index>=prices.size())stock_index=0;
        }
        else
        {
          while(true)
          {
            stock_index++;
            if(stock_index>=prices.size())stock_index=0;
            if(std::find(rsymbols.begin(),rsymbols.end(),stock_index) != rsymbols.end())break;
          }
        }
      }
      else
      {
        start_date_index = 4000;
        end_date_index = 0;
      }
      break;
    case 'w':if(!game_mode){end_date_index=0;start_date_index*=1.1f;if(start_date_index>4000)start_date_index=4000;}break;
    case 's':if(!game_mode){end_date_index=0;start_date_index/=1.1f;if(start_date_index<10)start_date_index=10;}break;
    case 'c':pick_start_index = true;break;
    case 'v':pick_end_index = true;break;
    case 'b':start_index=-1;end_index=-1;break;
    case 'z':
      buy_only = !buy_only;
      if(buy_only) 
      {
        if(scanner.buy.size()==0)
        {
          buy_only=false;
          stock_index++;
          if(stock_index>=prices.size())stock_index=0;
        }
        else
        {
          while(true)
          {
            stock_index++;
            if(stock_index>=prices.size())stock_index=0;
            if(scanner.buy.find(symbols[stock_index]) != scanner.buy.end())break;
          }
        }
      }
      break;
    case 27:exit(1);break;
    default:break;
  }
}

void passive_mouse(int x,int y)
{
  mouse_x = x;
  mouse_y = y;
}

void active_mouse(int x,int y)
{
  mouse_x = x;
  mouse_y = y;
}

int main(int argc,char ** argv)
{

  if(argc>=2)
  {
    learning_num = atoi(argv[1]);
  }

  if(argc>=3)
  {
    train_index = atoi(argv[2]);
  }

  if(argc>=4)
  {
    input_filename = std::string(argv[3]);
  }

  int seed;
  seed = time(0);
  srand(seed);

  bool awesome_macd = true;

  int synthetic_prices = (synthetic_range-1)*learning_offset;

  start_date_index = 4000;
  end_date_index = 0;
  
  // list all files in current directory.
  boost::filesystem::path p ("data");
  boost::filesystem::directory_iterator end_itr;
  // cycle through the directory
  int ind = 0;
  for (boost::filesystem::directory_iterator itr(p); itr != end_itr; ++itr,ind++)
  {
    // If it's not a directory, list it. If you want to list directories too, just remove this check.
    if (boost::filesystem::is_regular_file(itr->path())) {
      prices.push_back(std::vector<price>());
      // assign current file name to current_file and echo it out to the console.
      std::string current_file = itr->path().string();
      symbols.push_back(current_file);
      fprintf(stderr,"file:%s\n",current_file.c_str());
      read_data_yahoo(current_file,prices[ind],synthetic_prices);
    }
  }

  for(int i=0;i<prices.size();i++)
  {
    price::initialize_indicators(prices[i],awesome_macd);
  }

  scanner.scan(prices,symbols);

  for(int i=0;i<4;i++)
  {
    int sym = rand()%symbols.size();
    while(std::find(rsymbols.begin(),rsymbols.end(),sym)==rsymbols.end()){
      if(std::find(rsymbols.begin(),rsymbols.end(),sym)==rsymbols.end()){
        rsymbols.push_back(sym);
        break;
      }
      sym = rand()%symbols.size();
    }
  }

  std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
  for(int i=0;i<rsymbols.size();i++)
  {
    std::cout << symbols[rsymbols[i]] << std::endl;
  }
  std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;

  user = new User("Anton Kodochygov",10000,rsymbols);

   in_dump = new double[learning_num*prices.size()*robot-> get_input_size( input_learning_range)];
  out_dump = new double[learning_num*prices.size()*robot->get_output_size(output_learning_range)];
   in_test = new double[test_learning_num*prices.size()*robot-> get_input_size( input_learning_range)];
  out_test = new double[test_learning_num*prices.size()*robot->get_output_size(output_learning_range)];

  learning_samples = construct_learning_data(40+test_learning_num,40+test_learning_num+learning_num,learning_offset,in_dump,out_dump);
  test_learning_samples = construct_learning_data(0,test_learning_num,learning_offset,in_test,out_test);
  std::cout << "learning samples: " << learning_samples << std::endl;

  {
    if(training==false)
    {
      training=true;
      boost::thread * thr ( new boost::thread ( run_perceptron ) );
    }
  }

  glutInit(&argc, argv);
  glutInitWindowSize(width,height);
  glutInitWindowPosition(0,0);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow("stock bot");
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutPassiveMotionFunc(passive_mouse);
  glutMotionFunc(active_mouse);
  glutIdleFunc(idle);
  init();
  glutMainLoop();
  return 0;
}

