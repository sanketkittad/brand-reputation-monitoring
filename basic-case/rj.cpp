#include <iostream>
#include <vector>
using namespace std;

vector<int> memo;

int maxSecuritySum(vector<int>& security_val, int k,int pv,int i) {
    if (i >= security_val.size()) return 0;
    if (memo[i] != -1) return memo[i];

    int maxSum = INT_MIN;
   
    int sum = security_val[i] + maxSecuritySum(security_val, k,i, i + k);
    int npsum=INT_MIN;

    if(pv==-1){
        npsum=maxSecuritySum(security_val,k,-1,i+1);
    }
        
    memo[i] = max(sum,npsum);
    return memo[i];
}

int findStartNode(vector<int>& security_val, int k) {
    int n = security_val.size();
    memo.assign(n, -1);
    return maxSecuritySum(security_val, k,-1, 0);
}

int main() {
    vector<int> security_values = {3, 5, -2, -4, 9 ,16};
    int k = 2;
    int optimal_start_node = findStartNode(security_values, k);
    cout << "Optimal start node: " << optimal_start_node << endl;
    return 0;
}
