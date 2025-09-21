import numpy as np
import itertools
from ldpc import bposd_decoder
from bposd.css import css_code
import pickle
from scipy.sparse import coo_matrix
import relay_bp
from scipy.sparse import csr_matrix

# number of Monte Carlo trials
num_trials = 1

# Weight 7 faults, even with 100,000 BP iterations
# faulty_gates = [('CNOT', ('Xcheck', 0), ('data_left', 1)), ('CNOT', ('Xcheck', 6), ('data_right', 12)), ('CNOT', ('Xcheck', 62), ('data_right', 65)), ('CNOT', ('Xcheck', 67), ('data_right', 70)), ('CNOT', ('Xcheck', 3), ('data_right', 15)), ('CNOT', ('Xcheck', 10), ('data_right', 22)), ('CNOT', ('Xcheck', 13), ('data_right', 25))]
# faulty_gates = [('CNOT', ('Xcheck', 0), ('data_right', 6)), ('CNOT', ('Xcheck', 3), ('data_right', 0)), ('CNOT', ('Xcheck', 15), ('data_right', 12)), ('CNOT', ('Xcheck', 62), ('data_right', 65)), ('CNOT', ('Xcheck', 67), ('data_right', 70)), ('CNOT', ('Xcheck', 10), ('data_right', 22)), ('CNOT', ('Xcheck', 13), ('data_left', 31))]
# faulty_gates = [('CNOT', ('Xcheck', 10), ('data_left', 11)), ('CNOT', ('Xcheck', 0), ('data_right', 6)), ('CNOT', ('Xcheck', 15), ('data_right', 21)), ('CNOT', ('Xcheck', 3), ('data_right', 0)), ('CNOT', ('Xcheck', 62), ('data_right', 65)), ('CNOT', ('Xcheck', 67), ('data_right', 70)), ('CNOT', ('Xcheck', 13), ('data_right', 25))]

# Faults only with no more than ~1000 BP iterations
# faulty_gates = [('CNOT', ('Xcheck', 67), ('data_left', 68)), ('CNOT', ('Xcheck', 0), ('data_right', 6)), ('CNOT', ('Xcheck', 6), ('data_right', 12)), ('CNOT', ('Xcheck', 62), ('data_right', 65)), ('CNOT', ('Xcheck', 3), ('data_right', 15)), ('CNOT', ('Xcheck', 13), ('data_right', 25)), ('CNOT', ('Xcheck', 10), ('data_left', 28))]
# faulty_gates = [('CNOT', ('Xcheck', 0), ('data_left', 1)), ('CNOT', ('Xcheck', 5), ('data_right', 11)), ('CNOT', ('Xcheck', 3), ('data_right', 0)), ('CNOT', ('Xcheck', 62), ('data_right', 65)), ('CNOT', ('Xcheck', 10), ('data_right', 22)), ('CNOT', ('Xcheck', 13), ('data_right', 25)), ('CNOT', ('Xcheck', 67), ('data_right', 7))]

# Weight 8
# faulty_gates = [('CNOT', ('Xcheck', 10), ('data_left', 11)), ('CNOT', ('Xcheck', 13), ('data_left', 14)), ('CNOT', ('Xcheck', 67), ('data_left', 68)), ('CNOT', ('Xcheck', 0), ('data_right', 6)), ('CNOT', ('Xcheck', 15), ('data_right', 21)), ('CNOT', ('Xcheck', 62), ('data_right', 68)), ('CNOT', ('Xcheck', 3), ('data_right', 0)), ('CNOT', ('Xcheck', 6), ('data_right', 18))]

# Weight 6
# faulty_gates = [('CNOT', ('Xcheck', 27), ('data_left', np.int64(28))), ('CNOT', ('Xcheck', 33), ('data_left', np.int64(34))), ('CNOT', ('Xcheck', 48), ('data_left', np.int64(49))), ('CNOT', ('Xcheck', 24), ('data_right', np.int64(30))), ('CNOT', ('Xcheck', 39), ('data_right', np.int64(45))), ('CNOT', ('Xcheck', 21), ('data_right', np.int64(18)))]
# [('CNOT', ('Xcheck', 21), ('data_left', np.int64(22))), ('CNOT', ('Xcheck', 27), ('data_left', np.int64(28))), ('CNOT', ('Xcheck', 33), ('data_left', np.int64(34))), ('CNOT', ('Xcheck', 39), ('data_left', np.int64(40))), ('CNOT', ('Xcheck', 48), ('data_right', np.int64(54))), ('CNOT', ('Xcheck', 24), ('data_right', np.int64(27)))]
# [('CNOT', ('Xcheck', 21), ('data_left', np.int64(22))), ('CNOT', ('Xcheck', 27), ('data_left', np.int64(28))), ('CNOT', ('Xcheck', 33), ('data_left', np.int64(34))), ('CNOT', ('Xcheck', 39), ('data_left', np.int64(40))), ('CNOT', ('Xcheck', 48), ('data_left', np.int64(49))), ('CNOT', ('Xcheck', 24), ('data_right', np.int64(30)))]
# [('CNOT', ('Xcheck', 3), ('data_left', np.int64(4))), ('CNOT', ('Xcheck', 7), ('data_right', np.int64(13))), ('CNOT', ('Xcheck', 12), ('data_right', np.int64(18))), ('CNOT', ('Xcheck', 1), ('data_right', np.int64(13))), ('CNOT', ('Xcheck', 10), ('data_right', np.int64(22))), ('CNOT', ('Xcheck', 13), ('data_right', np.int64(25)))]
# [('CNOT', ('Xcheck', 3), ('data_left', np.int64(4))), ('CNOT', ('Xcheck', 13), ('data_left', np.int64(14))), ('CNOT', ('Xcheck', 9), ('data_right', np.int64(15))), ('CNOT', ('Xcheck', 16), ('data_right', np.int64(22))), ('CNOT', ('Xcheck', 4), ('data_right', np.int64(1))), ('CNOT', ('Xcheck', 7), ('data_right', np.int64(10)))]

# Weight 5
# faulty_gates = [('CNOT', ('Xcheck', 24), ('data_left', np.int64(25))), ('CNOT', ('Xcheck', 27), ('data_left', np.int64(28))), ('CNOT', ('Xcheck', 39), ('data_left', np.int64(40))), ('CNOT', ('Xcheck', 48), ('data_left', np.int64(49))), ('CNOT', ('Xcheck', 21), ('data_right', np.int64(27)))]
# faulty_gates = [('CNOT', ('Xcheck', 39), ('data_left', np.int64(40))), ('CNOT', ('Xcheck', 24), ('data_right', np.int64(30))), ('CNOT', ('Xcheck', 48), ('data_right', np.int64(54))), ('CNOT', ('Xcheck', 21), ('data_right', np.int64(18))), ('CNOT', ('Xcheck', 27), ('data_right', np.int64(39)))]
# faulty_gates = [('CNOT', ('Xcheck', 48), ('data_left', np.int64(49))), ('CNOT', ('Xcheck', 24), ('data_right', np.int64(30))), ('CNOT', ('Xcheck', 21), ('data_right', np.int64(18))), ('CNOT', ('Xcheck', 27), ('data_right', np.int64(39))), ('CNOT', ('Xcheck', 33), ('data_right', np.int64(45)))]
# Relay fails
# faulty_gates = [('CNOT', ('Xcheck', 21), ('data_left', np.int64(22))), ('CNOT', ('Xcheck', 27), ('data_left', np.int64(28))), ('CNOT', ('Xcheck', 48), ('data_left', np.int64(49))), ('CNOT', ('Xcheck', 24), ('data_right', np.int64(30))), ('CNOT', ('Xcheck', 33), ('data_right', np.int64(45)))]
# faulty_gates = [('CNOT', ('Xcheck', 24), ('data_left', np.int64(25))), ('CNOT', ('Xcheck', 27), ('data_left', np.int64(28))), ('CNOT', ('Xcheck', 33), ('data_left', np.int64(34))), ('CNOT', ('Xcheck', 39), ('data_right', np.int64(45))), ('CNOT', ('Xcheck', 21), ('data_right', np.int64(18)))]

# Weight 4
# faulty_gates = [('CNOT', ('Xcheck', 48), ('data_left', np.int64(49))), ('CNOT', ('Xcheck', 39), ('data_right', np.int64(45))), ('CNOT', ('Xcheck', 21), ('data_right', np.int64(33))), ('CNOT', ('Xcheck', 33), ('data_right', np.int64(45)))]
# faulty_gates = [('CNOT', ('Xcheck', 39), ('data_left', np.int64(40))), ('CNOT', ('Xcheck', 48), ('data_left', np.int64(49))), ('CNOT', ('Xcheck', 33), ('data_right', np.int64(30))), ('CNOT', ('Xcheck', 21), ('data_right', np.int64(33)))]
# faulty_gates = [('CNOT', ('Xcheck', 21), ('data_left', np.int64(22))), ('CNOT', ('Xcheck', 27), ('data_left', np.int64(28))), ('CNOT', ('Xcheck', 24), ('data_right', np.int64(36))), ('CNOT', ('Xcheck', 33), ('data_right', np.int64(45)))]
# faulty_gates = [('CNOT', ('Xcheck', 48), ('data_left', np.int64(49))), ('CNOT', ('Xcheck', 24), ('data_right', np.int64(27))), ('CNOT', ('Xcheck', 27), ('data_right', np.int64(39))), ('CNOT', ('Xcheck', 33), ('data_right', np.int64(45)))]
# faulty_gates = [('CNOT', ('Xcheck', 39), ('data_left', np.int64(40))), ('CNOT', ('Xcheck', 48), ('data_left', np.int64(49))), ('CNOT', ('Xcheck', 33), ('data_right', np.int64(30))), ('CNOT', ('Xcheck', 21), ('data_right', np.int64(33)))]
# faulty_gates = [('CNOT', ('Xcheck', 48), ('data_left', np.int64(49))), ('CNOT', ('Xcheck', 39), ('data_right', np.int64(45))), ('CNOT', ('Xcheck', 21), ('data_right', np.int64(33))), ('CNOT', ('Xcheck', 33), ('data_right', np.int64(45)))]

# Weight 4 - Relay fails
faulty_gates = [('CNOT', ('Xcheck', 21), ('data_left', np.int64(22))), ('CNOT', ('Xcheck', 27), ('data_left', np.int64(28))), ('CNOT', ('Xcheck', 24), ('data_right', np.int64(30))), ('CNOT', ('Xcheck', 33), ('data_right', np.int64(45)))]


# faulty_gates = []
error_rate = 0.001  # Override below for actual error rate!
relay_decoder = "RelayDecoder"  # RelayDecoderF64 MinSumBPDecoderF64

# code parameters and number of syndrome cycles
n = 144
k = 12
d = 12
num_cycles = 12

# load decoder data from file (must be created with decoder_setup.py)
title = './TMP/mydata_' + str(n) + '_' + str(k) + '_p_' + str(error_rate) + '_cycles_' + str(num_cycles)
print('reading data from file')
print(title)
with open(title, 'rb') as fp:
	mydata = pickle.load(fp)

error_rate = 0.0  # Override of the actual error rate!

# file to save simulation results
fname = './CODE_' + str(n) + '_' + str(k) + '_' + str(d) + '/result'

# format of the result file
# column 1: error rate
# column 2: number of syndrome cycles
# column 3: number of Monte Carlo trials 
# column 4: number of Monte Carlo trials that resulted in a logical error


HdecX = mydata['HdecX']
HdecZ = mydata['HdecZ']
channel_probsX = mydata['probX']
channel_probsZ = mydata['probZ']
lin_order = mydata['lin_order']
assert(mydata['num_cycles']==num_cycles)
data_qubits = mydata['data_qubits']
Xchecks=mydata['Xchecks']
Zchecks=mydata['Zchecks']
cycle = mydata['cycle']
HX = mydata['HX']
HZ = mydata['HZ']
lx = mydata['lx']
lz = mydata['lz']
first_logical_rowZ=mydata['first_logical_rowZ']
first_logical_rowX=mydata['first_logical_rowX']
ell=mydata['ell']
m=mydata['m']
a1=mydata['a1']
a2=mydata['a2']
a3=mydata['a3']
b1=mydata['b1']
b2=mydata['b2']
b3=mydata['b3']
sX=mydata['sX']
sZ=mydata['sZ']
# assert(error_rate==mydata['error_rate'])
cycle_repeated = num_cycles*cycle

# setup BP-OSD decoder parameters
my_bp_method = "ms"
if relay_decoder == "":
	my_max_iter = 10000
	my_osd_method = "osd_cs"
	my_osd_order = 10
	my_ms_scaling_factor = 0
else:
	my_max_iter = 120



# code length
n = 2*m*ell

n2 = m*ell




def generate_noisy_circuit(p):
	error_rate_meas = p
	error_rate_idle = p
	error_rate_init = p
	error_rate_cnot = p
	circ = []
	err_cnt=0
	for gate in cycle_repeated:
		assert(gate[0] in ['CNOT','IDLE','PrepX','PrepZ','MeasX','MeasZ'])
		if gate[0]=='MeasX':
			if np.random.uniform()<=error_rate_meas:
				circ.append(('Z',gate[1]))
				err_cnt+=1
			circ.append(gate)
			continue
		if gate[0]=='IDLE':
			if np.random.uniform()<=error_rate_idle:
				ptype = np.random.randint(3)
				if ptype==0:
					circ.append(('X',gate[1]))
				if ptype==1:
					circ.append(('Y',gate[1]))
				if ptype==2:
					circ.append(('Z',gate[1]))
				err_cnt+=1
			continue
		if gate[0]=='PrepX':
			circ.append(gate)
			if np.random.uniform()<=error_rate_init:
				circ.append(('Z',gate[1]))
				err_cnt+=1
			continue
		if gate[0]=='CNOT':
			circ.append(gate)
			if gate in faulty_gates:
				circ.append(('X', gate[1]))  # These are X checks, always CNOT controls
				faulty_gates.remove(gate)  # To make sure every error occurs only once
				err_cnt += 1
				continue
			if np.random.uniform()<=error_rate_cnot:
				error_type = np.random.randint(15)
				if error_type==0:
					circ.append(('X',gate[1]))
					err_cnt+=1
					continue
				if error_type==1:
					circ.append(('Y',gate[1]))
					err_cnt+=1
					continue
				if error_type==2:
					circ.append(('Z',gate[1]))
					err_cnt+=1
					continue
				if error_type==3:
					circ.append(('X',gate[2]))
					err_cnt+=1
					continue
				if error_type==4:
					circ.append(('Y',gate[2]))
					err_cnt+=1
					continue
				if error_type==5:
					circ.append(('Z',gate[2]))
					err_cnt+=1
					continue
				if error_type==6:
					circ.append(('XX',gate[1],gate[2]))
					err_cnt+=1
					continue
				if error_type==7:
					circ.append(('YY',gate[1],gate[2]))
					err_cnt+=1
					continue
				if error_type==8:
					circ.append(('ZZ',gate[1],gate[2]))
					err_cnt+=1
					continue
				if error_type==9:
					circ.append(('XY',gate[1],gate[2]))
					err_cnt+=1
					continue
				if error_type==10:
					circ.append(('YX',gate[1],gate[2]))
					err_cnt+=1
					continue
				if error_type==11:
					circ.append(('YZ',gate[1],gate[2]))
					err_cnt+=1
					continue
				if error_type==12:
					circ.append(('ZY',gate[1],gate[2]))
					err_cnt+=1
					continue
				if error_type==13:
					circ.append(('XZ',gate[1],gate[2]))
					err_cnt+=1
					continue
				if error_type==14:
					circ.append(('ZX',gate[1],gate[2]))
					err_cnt+=1
					continue
		if gate[0]=='PrepZ':
			circ.append(gate)
			if np.random.uniform()<=error_rate_init:
				circ.append(('X',gate[1]))
				err_cnt+=1
			continue
		if gate[0]=='MeasZ':
			if np.random.uniform()<=error_rate_meas:
				circ.append(('X',gate[1]))
				err_cnt+=1
			circ.append(gate)
			continue

	print(f"err_cnt = {err_cnt}")
	return circ




# we only look at the action of the circuit on Z errors; 0 means no error, 1 means error
def simulate_circuitZ(C):
	syndrome_history = []
	# keys = Xchecks, vals = list of positions in the syndrome history array
	syndrome_map = {}
	state = np.zeros(2*n,dtype=int)
	# need this for debugging
	err_cnt = 0
	syn_cnt = 0
	for gate in C:
		if gate[0]=='CNOT':
			assert(len(gate)==3)
			control = lin_order[gate[1]]
			target = lin_order[gate[2]]
			state[control] = (state[target] + state[control]) % 2
			continue
		if gate[0]=='PrepX':
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q]=0
			continue
		if gate[0]=='MeasX':
			assert(len(gate)==2)
			assert(gate[1][0]=='Xcheck')
			q = lin_order[gate[1]]
			syndrome_history.append(state[q])
			if gate[1] in syndrome_map:
				syndrome_map[gate[1]].append(syn_cnt)
			else:
				syndrome_map[gate[1]] = [syn_cnt]
			syn_cnt+=1
			continue
		if gate[0] in ['Z','Y']:
			err_cnt+=1
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['ZX', 'YX']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['XZ','XY']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[2]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['ZZ','YY','YZ','ZY']:
			err_cnt+=1
			assert(len(gate)==3)
			q1 = lin_order[gate[1]]
			q2 = lin_order[gate[2]]
			state[q1] = (state[q1] + 1) % 2
			state[q2] = (state[q2] + 1) % 2
			continue
	return np.array(syndrome_history,dtype=int),state,syndrome_map,err_cnt


# we only look at the action of the circuit on X errors; 0 means no error, 1 means error
def simulate_circuitX(C):
	syndrome_history = []
	# keys = Zchecks, vals = list of positions in the syndrome history array
	syndrome_map = {}
	state = np.zeros(2*n,dtype=int)
	# need this for debugging
	err_cnt = 0
	syn_cnt = 0
	for gate in C:
		if gate[0]=='CNOT':
			assert(len(gate)==3)
			control = lin_order[gate[1]]
			target = lin_order[gate[2]]
			state[target] = (state[target] + state[control]) % 2
			continue
		if gate[0]=='PrepZ':
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q]=0
			continue
		if gate[0]=='MeasZ':
			assert(len(gate)==2)
			assert(gate[1][0]=='Zcheck')
			q = lin_order[gate[1]]
			syndrome_history.append(state[q])
			if gate[1] in syndrome_map:
				syndrome_map[gate[1]].append(syn_cnt)
			else:
				syndrome_map[gate[1]] = [syn_cnt]
			syn_cnt+=1
			continue
		if gate[0] in ['X','Y']:
			err_cnt+=1
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['XZ', 'YZ']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['ZX','ZY']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[2]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['XX','YY','XY','YX']:
			err_cnt+=1
			assert(len(gate)==3)
			q1 = lin_order[gate[1]]
			q2 = lin_order[gate[2]]
			state[q1] = (state[q1] + 1) % 2
			state[q2] = (state[q2] + 1) % 2
			continue
	return np.array(syndrome_history,dtype=int),state,syndrome_map,err_cnt





# begin decoding
if relay_decoder == "MinSumBPDecoderF64":
	print(f"Using Relay decoder class: MinSumBPDecoderF64")

	x_bpd2 = relay_bp.MinSumBPDecoderF64(
		csr_matrix(HdecX),
		error_priors=np.asarray(channel_probsX),  # Set the priors probability for each error
		gamma0=0.125,  # Uniform memory weight for the first ensemble
		max_iter=my_max_iter,  # Max BP iterations for the first ensemble
	)
	z_bpd2 = relay_bp.MinSumBPDecoderF64(
		csr_matrix(HdecZ),
		error_priors=np.asarray(channel_probsZ),  # Set the priors probability for each error
		gamma0=0.125,  # Uniform memory weight for the first ensemble
		max_iter=my_max_iter,  # Max BP iterations for the first ensemble
	)
elif relay_decoder == "RelayDecoder":
	print(f"Using Relay decoder class: RelayDecoderF64")

	x_bpd2 = relay_bp.RelayDecoderF64(
		csr_matrix(HdecX),
		error_priors=np.asarray(channel_probsX),  # Set the priors probability for each error
		gamma0=0.125,  # Uniform memory weight for the first ensemble
		pre_iter=80,  # Max BP iterations for the first ensemble
		num_sets=601,  # Number of relay ensemble elements
		set_max_iter=int(my_max_iter / 2),  # Max BP iterations per relay ensemble
		gamma_dist_interval=(-0.24, 0.66),
		# Set the uniform distribution range for disordered memory weight selection
		stop_nconv=5,  # Number of relay solutions to find before stopping (the best will be selected)
	)
	z_bpd2 = relay_bp.RelayDecoderF64(
		csr_matrix(HdecZ),
		error_priors=np.asarray(channel_probsZ),  # Set the priors probability for each error
		gamma0=0.125,  # Uniform memory weight for the first ensemble
		pre_iter=80,  # Max BP iterations for the first ensemble
		num_sets=601,  # Number of relay ensemble elements
		set_max_iter=int(my_max_iter / 2),  # Max BP iterations per relay ensemble
		gamma_dist_interval=(-0.24, 0.66),
		# Set the uniform distribution range for disordered memory weight selection
		stop_nconv=5,  # Number of relay solutions to find before stopping (the best will be selected)
	)

else:
	print(f"Using BP+OSD-CS decoder class: ldpc.bposd_decoder")

	bpdX=bposd_decoder(
		HdecX,#the parity check matrix
		channel_probs=channel_probsX, #assign error_rate to each qubit. This will override "error_rate" input variable
		max_iter=my_max_iter, #the maximum number of iterations for BP)
		bp_method=my_bp_method,
		ms_scaling_factor=my_ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
		osd_method=my_osd_method, #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
		osd_order=my_osd_order #the osd search depth
		)


	bpdZ=bposd_decoder(
		HdecZ,#the parity check matrix
		channel_probs=channel_probsZ, #assign error_rate to each qubit. This will override "error_rate" input variable
		max_iter=my_max_iter, #the maximum number of iterations for BP)
		bp_method=my_bp_method,
		ms_scaling_factor=my_ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
		osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
		osd_order=my_osd_order #the osd search depth
		)


good_trials=0
bad_trials=0
for trial in range(num_trials):

	circ = generate_noisy_circuit(error_rate)

	# error correction result
	# True = success
	# False = fail
	ec_resultZ = False
	ec_resultX = False
	
	# correct Z errors 
	syndrome_history,state,syndrome_map,err_cntZ = simulate_circuitZ(circ+cycle+cycle)
	assert(len(syndrome_history)==n2*(num_cycles+2))
	state_data_qubits = [state[lin_order[q]] for q in data_qubits]
	syndrome_final_logical = (lx @ state_data_qubits) % 2
	# apply syndrome sparsification map
	syndrome_history_copy = syndrome_history.copy()
	for c in Xchecks:
		pos = syndrome_map[c]
		assert(len(pos)==(num_cycles+2))
		for row in range(1,num_cycles+2):
			syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
	syndrome_history%= 2
	assert(HdecZ.shape[0]==len(syndrome_history))
	if relay_decoder != "":
		low_weight_error = z_bpd2.decode(np.asarray(syndrome_history, dtype=np.uint8))
	else:
		bpdZ.decode(syndrome_history)
		low_weight_error = bpdZ.osdw_decoding

	assert(len(low_weight_error)==HZ.shape[1])
	syndrome_history_augmented_guessed = (HZ @ low_weight_error) % 2
	syndrome_final_logical_guessed = syndrome_history_augmented_guessed[first_logical_rowZ:(first_logical_rowZ+k)]
	ec_resultZ = np.array_equal(syndrome_final_logical_guessed,syndrome_final_logical)
	
	
	if ec_resultZ:
		# correct X errors 
		syndrome_history,state,syndrome_map,err_cntX = simulate_circuitX(circ+cycle+cycle)
		assert(len(syndrome_history)==n2*(num_cycles+2))
		state_data_qubits = [state[lin_order[q]] for q in data_qubits]
		syndrome_final_logical = (lz @ state_data_qubits) % 2
		# apply syndrome sparsification map
		syndrome_history_copy = syndrome_history.copy()
		for c in Zchecks:
			pos = syndrome_map[c]
			assert(len(pos)==(num_cycles+2))
			for row in range(1,num_cycles+2):
				syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
		syndrome_history%= 2
		assert(HdecX.shape[0]==len(syndrome_history))
		if relay_decoder != "":
			low_weight_error = x_bpd2.decode(np.asarray(syndrome_history, dtype=np.uint8))
		else:
			bpdX.decode(syndrome_history)
			low_weight_error = bpdX.osdw_decoding

		assert(len(low_weight_error)==HX.shape[1])
		syndrome_history_augmented_guessed = (HX @ low_weight_error) % 2
		syndrome_final_logical_guessed = syndrome_history_augmented_guessed[first_logical_rowX:(first_logical_rowX+k)]
		ec_resultX = np.array_equal(syndrome_final_logical_guessed,syndrome_final_logical)
		
	

	if ec_resultZ and ec_resultX:
		good_trials+=1
	else:
		bad_trials+=1
		
	assert((trial+1)==(good_trials+bad_trials))

	print(str(error_rate) + '\t' + str(num_cycles) + '\t' + str(trial+1) + '\t' + str(bad_trials))
	

assert(num_trials==(good_trials+bad_trials))

print(str(error_rate) + '\t' + str(num_cycles) + '\t' + str(num_trials) + '\t' + str(bad_trials),file=open(fname,'a'))
