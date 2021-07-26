import os
import queue
import multiprocessing as mp
import subprocess as sp


# 1. How many workers do you want for each GPU?
# ---------------------------------------------
# E.g. you have 2 RTX 2080 (8GB) and 2 RTX 2080 Ti (11GB),
# configure the number of processes to be run in parallel for each GPU.
CUDA_WORKERS = {
    0: 1, # 4 workers for CUDA_VISIBLE_DEVICES=0
    #1: 4, # 4 workers for CUDA_VISIBLE_DEVICES=1
    #2: 4, # 4 workers for CUDA_VISIBLE_DEVICES=2
    #3: 4, # 4 workers for CUDA_VISIBLE_DEVICES=3 
}

# 2. Use list comprehension to generate the training commands with 
# different hyperparameter combination
COMMANDS = [
    ['python', 'train_dpm.py',
     '--num_steps', str(500),
     '--n_episode', str(e),
     '--episode_step', str(s),
     '--lr', str(lr),
     '--win_size', str(31),
     '--reg_w', str(reg),
     '--cash_reward', 'custom',
     '--cr_scale', str(m_scale),
     '--cr_shift', str(m_shift),
     '--rolling',
     '--num_rolling_steps', str(rs)
     ] 
    # argv list
    for e, s in [(128, 50)]
    for lr in [3e-6]
    for rs in [20]
    for pr in [5e-2]
    for reg in [1e-4]
    for m_shift, m_scale in [(1.004, 5)]
]


def worker(cuda_no, worker_no, cmd_queue):
    worker_name = 'CUDA-{}:{}'.format(cuda_no, worker_no)
    print(worker_name, 'started')
    
    env = os.environ.copy()
    # overwrite visible cuda devices
    env['CUDA_VISIBLE_DEVICES'] = str(cuda_no)
    
    while True:
        cmd = cmd_queue.get()
        
        if cmd is None:
            cmd_queue.task_done()
            break
        
        print(worker_name, cmd)
        
        shell = {str: True, list: False}.get(type(cmd))
        assert shell is not None, 'cmd should be list or str'
        
        sp.Popen(cmd, shell=shell, env=env).wait()
        cmd_queue.task_done()
    
    print(worker_name, 'stopped')


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    cmd_queue = mp.JoinableQueue()
    
    for cmd in COMMANDS:
        cmd_queue.put(cmd)
        
    for _ in range(sum(CUDA_WORKERS.values())):
        # workers stop after getting None
        cmd_queue.put(None)
        
    procs = [
        mp.Process(target=worker, args=(cuda_no, worker_no, cmd_queue), daemon=True)
        for cuda_no, num_workers in CUDA_WORKERS.items()
        for worker_no in range(num_workers)
    ]
    
    for proc in procs:
        proc.start()

    cmd_queue.join()
        
    for proc in procs:
        proc.join()