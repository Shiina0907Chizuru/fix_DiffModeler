from training.train_utils import AverageMeter,ProgressMeter
import time
import os

def train_diffmodeler(train_loader,ddim_runner,model,
                                       epoch,train_log_path,params):
    model.train()
    avg_meters = {'data_time':AverageMeter('data_time'),
                  'train_time':AverageMeter('train_time'),
                  'iou':AverageMeter('iou'),
                  'loss': AverageMeter('loss'),}
    
    progress = ProgressMeter(
        len(train_loader),
        [avg_meters['data_time'],
         avg_meters['train_time'],
         avg_meters['iou'],
         avg_meters['loss']]
        ,prefix="Epoch: [{}]".format(epoch))
    end_time=time.time()
    batch_size = params['train']['batch_size']
    
    # 检查是否开启问题批次监控
    trouble_log = params.get('trouble_log', False)
    trouble_log_path = params.get('trouble_log_path', None)
    
    # 如果启用问题批次监控但未指定路径，使用默认路径
    if trouble_log and not trouble_log_path:
        # 获取log_path，这是由main_worker生成的日志目录
        log_path = os.path.dirname(train_log_path)
        trouble_log_dir = os.path.join(log_path, "trouble_logs")
        os.makedirs(trouble_log_dir, exist_ok=True)
        trouble_log_path = os.path.join(trouble_log_dir, "train_trouble_log.txt")
        print(f"未找到问题批次监控日志路径，使用默认路径: {trouble_log_path}")
        
        # 确保日志文件存在并包含标题
        if not os.path.exists(trouble_log_path):
            with open(trouble_log_path, 'w') as f:
                f.write("=== DiffModeler Training Trouble Log ===\n")
                f.write(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Epoch: {epoch}\n\n")

    for batch_idx,data in enumerate(train_loader):
        ddim_runner.feed_data(data)
        avg_meters['data_time'].update(time.time()-end_time,batch_size)
        loss_dict = ddim_runner.optimize_parameters(trouble_log=trouble_log, log_path=trouble_log_path, batch_idx=batch_idx)
        avg_meters['loss'].update(loss_dict['loss'], batch_size)
        avg_meters['iou'].update(loss_dict['iou'], batch_size)
        avg_meters['train_time'].update(time.time()-end_time,batch_size )
        end_time = time.time()
        progress.display(batch_idx)
        progress.write_record(batch_idx,train_log_path)
    return avg_meters['loss'].avg
