#include <iostream>
#include "save_and_load.cpp"
using namespace std;

vector<tuple<vector<vector<double>>, vector<vector<double>>>> get_parameters(vector<Layer_Dense>& layer_params){
    vector<tuple<vector<vector<double>>, vector<vector<double>>>> parameters;
    for(int i = 0; i < layer_params.size(); i++)
        parameters.push_back(layer_params[i].get_parameters());
    return parameters;
}

void set_parameters(vector<Layer_Dense*>& layer_params, vector<tuple<vector<vector<double>>, vector<vector<double>>>>& parameters){
    for(int i = 0; i < layer_params.size(); i++)
        layer_params[i]->set_parameters(parameters[i]);
}