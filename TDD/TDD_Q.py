import numpy as np
from TDD.TN import Index,Tensor,TensorNetwork
from qiskit.quantum_info.operators import Operator
import time

def is_diagonal(U):
    i, j = np.nonzero(U)
    return np.all(i == j)

def add_hyper_index(var_list,hyper_index):
    for var in var_list:
        if not var in hyper_index:
            hyper_index[var]=0
            
def reshape(U):
    if U.shape==(1,1):
        return U
    
    if U.shape[0]==U.shape[1]:
        split_U=np.split(U,2,1)
    else:
        split_U=np.split(U,2,0)
    split_U[0]=reshape(split_U[0])
    split_U[1]=reshape(split_U[1]) 
    return np.array([split_U])[0]            
            
def get_real_qubit_num(cir):
    """Calculate the real number of qubits of a circuit"""
    gates=cir.data
    q=0
    for k in range(len(gates)):
        q=max(q,max([qbit._index for qbit in gates[k][1]]))
    return q+1

def cir_2_tn(cir):
    """return the dict that link every quantum gate to the corresponding index"""
#     print(1)
#     t=time.time()
    
    
    hyper_index=dict()
    qubits_index = dict()
    start_tensors= dict()
    end_tensors = dict()
    
    qubits_num=get_real_qubit_num(cir)

    for k in range(qubits_num):
        qubits_index[k]=0
        
    tn=TensorNetwork([],tn_type='cir',qubits_num=qubits_num)
                
    gates=cir.data
    for k in range(len(gates)):
        g=gates[k]
        nam=g[0].name
        q = [q._index for q in g[1]]
        q.reverse()
        var=[]

        ts=Tensor([],[],nam,q)
        
        if nam=='reset':
            continue

        U=Operator(g[0]).data
        
        for k in q:
            var_in='x'+ str(k)+'_'+str(qubits_index[k])
            var_out='x'+ str(k)+'_'+str(qubits_index[k]+1)
            add_hyper_index([var_in,var_out],hyper_index)
            var+=[Index(var_in,hyper_index[var_in]),Index(var_out,hyper_index[var_out])]
            if qubits_index[k]==0 and hyper_index[var_in]==0:
                start_tensors[k]=ts
            end_tensors[k]=ts                
            qubits_index[k]+=1
        if len(q)>1:
            U=reshape(U)
            
        if len(q)==1:
            U=U.T
        ts.data=U
        ts.index_set=var
        tn.tensors.append(ts)
        
#         for k in ts.index_set:
#             print(k)
#         print(ts.data)         

    for k in range(qubits_num):
        if k in start_tensors:
            last1=Index('x'+str(k)+'_'+str(0),0)
            new1=Index('x'+str(k),0)            
            start_tensors[k].index_set[start_tensors[k].index_set.index(last1)]=new1
        if k in end_tensors:
            last2=Index('x'+str(k)+'_'+str(qubits_index[k]),hyper_index['x'+str(k)+'_'+str(qubits_index[k])])
            new2=Index('y'+str(k),0)            
            end_tensors[k].index_set[end_tensors[k].index_set.index(last2)]=new2
               
    for k in range(qubits_num):
        U=np.eye(2)
        if qubits_index[k]==0 and not 'x'+str(k)+'_'+str(0) in hyper_index:
            var_in='x'+str(k)
            var=[Index('x'+str(k),0),Index('y'+str(k),0)]
            ts=Tensor(U,var,'nu_q',[k])
            tn.tensors.append(ts)            
    
    all_indexs=[]
    for k in range(qubits_num):
        all_indexs.append('x'+str(k))
        for k1 in range(qubits_index[k]+1):
            all_indexs.append('x'+str(k)+'_'+str(k1))
        all_indexs.append('y'+str(k))
#     print(4)
#     print(time.time()-t)
    return tn,all_indexs

def add_inputs(tn,input_s,qubits_num):
    U0=np.array([1,0])
    U1=np.array([0,1])
    U_p=1/np.sqrt(2)*np.array([1,1])
    U_m=1/np.sqrt(2)*np.array([1,-1])
    U_i=1/np.sqrt(2)*np.array([1,1j])
    U_t=1/np.sqrt(2)*np.array([1,1/np.sqrt(2)+1/np.sqrt(2)*1j])
    if len(input_s)!= qubits_num:
        print("inputs is not match qubits number")
        return 
    for k in range(qubits_num-1,-1,-1):
        if input_s[k]==0:
            ts=Tensor(U0,[Index('x'+str(k))],'in',[k])
        elif input_s[k]==1:
            ts=Tensor(U1,[Index('x'+str(k))],'in',[k])
        elif input_s[k]=='+':
            ts=Tensor(U_p,[Index('x'+str(k))],'in',[k])      
        elif input_s[k]=='-':
            ts=Tensor(U_m,[Index('x'+str(k))],'in',[k])
        elif input_s[k]=='i':
            ts=Tensor(U_i,[Index('x'+str(k))],'in',[k])
        elif input_s[k]=='t':
            ts=Tensor(U_t,[Index('x'+str(k))],'in',[k])                 
        else:
            print('Only support computational basis input')
        tn.tensors.insert(0,ts)
            
def add_outputs(tn,output_s,qubits_num):
    U0=np.array([1,0])
    U1=np.array([0,1])
    # if len(output_s)!= qubits_num:
    #     print("outputs is not match qubits number")
    #     return 
    for k in output_s:
        if output_s[k]==0:
            ts=Tensor(U0,[Index('y'+str(k))],'out',[k])
        elif output_s[k]==1:
            ts=Tensor(U1,[Index('y'+str(k))],'out',[k])
        else:
            print('Only support computational basis output')
        tn.tensors.append(ts)       

def add_trace_line(tn,qubits_num):
    U=np.eye(2)
    for k in range(qubits_num-1,-1,-1):
        var_in='x'+str(k)
        var=[Index('x'+str(k),0),Index('y'+str(k),0)]
        ts=Tensor(U,var,'tr',[k])
        tn.tensors.insert(0,ts)
        
    

def gen_cir(name=None,qubit_num = 1,gate_num = 1):
    from qiskit import QuantumCircuit
    import random
    cir=QuantumCircuit(qubit_num)
    
    if name=='Random_Clifford':
        gate_set = ['x','y','z','h','s','cx']
        
        for k in range(gate_num):
            g = gate_set[random.randint(0,len(gate_set)-1)]
            q = random.randint(0,qubit_num-1)
            if g=='cx':
                q2 = random.randint(0,qubit_num-1)
                while q2==q:
                    q2 = random.randint(0,qubit_num-1)
                eval('cir.'+g+str(tuple([q,q2])))
            else:
                eval('cir.'+g+str(tuple([q])))
                
        return cir
    
    if name=='Random_Clifford_T':
        gate_set = ['x','y','z','h','s','cx','t']
        
        for k in range(gate_num):
            g = gate_set[random.randint(0,len(gate_set)-1)]
            q = random.randint(0,qubit_num-1)
            if g=='cx':
                q2 = random.randint(0,qubit_num-1)
                while q2==q:
                    q2 = random.randint(0,qubit_num-1)
                eval('cir.'+g+str(tuple([q,q2])))
            else:
                eval('cir.'+g+str(tuple([q])))
                
        return cir    
    