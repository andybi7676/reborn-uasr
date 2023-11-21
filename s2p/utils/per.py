import os
from tqdm import tqdm

# ref_f = "/home/b07502072/u-speech2speech/s2p/utils/asr_test.phones.txt"
# hyp_f = "/home/b07502072/u-speech2speech/s2p/multirun/2022-04-09/14-52-16/0/asr_test/asr_test.txt"
ref_f = "/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_fr/train_70h/skhc/asr_test.phones.txt"
hyp_f = "/home/b07502072/u-speech2speech/s2p/multirun/cv4_fr/xlsr/asr_test.phones.txt"
# reduce_set = [chr(771), chr(720)]
reduce_set = []

def read_phn_file(file_path):
    data = []
    with open(file_path, 'r') as fr:
        for line in fr:
            for re in reduce_set:
                line = line.replace(re, "")
            line = line.strip().split()
            data.append(line)
    return data

def cal_per(ref_data, hyp_data):
    import edit_distance
    S, D, I, N = (0, 0, 0, 0)
    count = 0
    for ref, hyp in tqdm(zip(ref_data, hyp_data)):
        if count < 5:
            print(' '.join(hyp))
            print(' '.join(ref))
        sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
        opcodes = sm.get_opcodes()
        
        # Substitution
        s = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'replace' else 0) for x in opcodes])
        # Deletion
        d = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'delete' else 0) for x in opcodes])
        # Insertion
        i = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'insert' else 0) for x in opcodes])
        n = len(ref)
        
        S += s
        D += d
        I += i
        N += n
        count += 1
    return S, D, I, N, count

def main():
    ref_data = read_phn_file(ref_f)
    hyp_data = read_phn_file(hyp_f)
    S, D, I, N, count = cal_per(ref_data, hyp_data) 
    
    print(f"PER: {(S+D+I)/N *100} %")
    print(f"DEL RATE: {(D)/N *100} %")
    print(f"INS RATE: {(I)/N *100} %")
    print(f"SUB RATE: {(S)/N *100} %")
    print(count)

if __name__ == "__main__":
    main()