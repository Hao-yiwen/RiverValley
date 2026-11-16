"""将训练日志汇总到同一个 SwanLab 在线面板，监控故障不进入训练主进程。"""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from colab_overnight import effective_status

logger = logging.getLogger(__name__)
TOTAL_PRETRAIN_STEPS = 7629
TOKENS_PER_UPDATE = 131072
TOTAL_SFT_STEPS = 324


def map_training_record(stage, row):
    """明确区分训练损失、验证损失和两阶段步数，避免把 SFT 当成预训练继续计数。"""
    prefix = '预训练' if stage == 'pretrain' else 'SFT'
    step = row.get('step')
    if not isinstance(step, int) or step < 0:
        return {}
    result = {f'{prefix}/step': step}
    names = {'loss': '训练loss', 'grad_norm': '梯度范数', 'lr': '学习率',
             'test': '验证loss', 'val_zh': '中文验证loss', 'val_en': '英文验证loss'}
    for source, label in names.items():
        value = row.get(source)
        if isinstance(value, (int, float)) and math.isfinite(value):
            result[f'{prefix}/{label}'] = value
    if stage == 'pretrain':
        result['进度/预训练完成百分比'] = min(100, step / TOTAL_PRETRAIN_STEPS * 100)
        result['进度/已训练token数'] = step * TOKENS_PER_UPDATE
        seconds = row.get('update_seconds')
        if isinstance(seconds, (float, int)) and math.isfinite(seconds) and seconds > 0:
            result['性能/单步秒数'] = seconds
            result['性能/计算token每秒'] = TOKENS_PER_UPDATE / seconds
    else:
        result['进度/SFT完成百分比'] = min(100, step / TOTAL_SFT_STEPS * 100)
    return result


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', default='/content/river-valley-run')
    parser.add_argument('--persistent-root', required=True)
    parser.add_argument('--credential-file', default='/content/river_valley_monitor.env')
    parser.add_argument('--interval', type=float, default=15)
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError('监控间隔必须大于零')
    root, persistent = Path(args.run_root), Path(args.persistent_root)
    credential = Path(args.credential_file)
    credential.chmod(0o600)
    for line in credential.read_text(encoding='utf-8').splitlines():
        if line.startswith('SWANLAB_API_KEY='):
            os.environ['SWANLAB_API_KEY'] = line.split('=', 1)[1]
    import swanlab
    swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=False)
    run = swanlab.init(project='river-valley-1b-training', experiment_name='1B 双语首阶段 · 2026-10-11',
        id='rv1b-bilingual-20261011', resume='allow', public=False, mode='online',
        logdir=str(root / 'monitor_sdk'),
        description='从零训练约 1.01B 参数；预训练约 10 亿 token → SFT → 保存并释放 GPU。2026-10-18 用户解除旧 85 CU 上限，要求完成全程；问答测试由用户本地进行。',
        config={'模型参数量':1012488192,'预训练目标step':TOTAL_PRETRAIN_STEPS,'预训练目标token':999948288,
                'batch_size':16,'accumulation_steps':8,'sequence_length':1024,'SFT目标step':TOTAL_SFT_STEPS,
                '预算策略':'完成预训练和 SFT；旧 85 CU 限制已解除','GPU':'A100-SXM4-80GB','当前CU每小时':6.77,
                'Drive备份目录':'RiverValley/first_training_20261011','刷新说明':'资源每 15 秒采样；训练记录每 20 步约 85 秒；验证每 200 步；CU 账单每分钟同步，陈旧时标注读数延迟。'})
    for prefix in ('预训练','SFT'):
        run.define_metric(prefix+'/*',x_axis=prefix+'/step',section_name=prefix+' · loss 与稳定性')
        run.define_metric(prefix+'/step',hidden=True)
    # 当前网页与 SDK 的内置相对时间轴配置不一致，显式记录分钟轴确保图表可读。
    run.define_metric('监控/已运行分钟',hidden=True)
    for prefix in ('进度','性能','资源','成本','时间','备份'):
        run.define_metric(prefix+'/*',x_axis='监控/已运行分钟',section_name=prefix)
    link={'url':run.url,'id':run.id,'project':'river-valley-1b-training','created_at':time.time(),'swanlab_version':swanlab.__version__}
    for path in (root/'monitor_url.json',persistent/'monitor_url.json'):
        path.write_text(json.dumps(link,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    logger.info('在线监控已创建：%s',run.url)
    previous=read_json(root/'monitor_heartbeat.json')
    positions={stage:previous.get('positions',{}).get(stage,0) for stage in ('pretrain','sft')}
    steps={stage:previous.get('steps',{}).get(stage,0) for stage in positions}
    durations=[]
    for line in (root/'checkpoints/pretrain/metrics.jsonl').read_text(encoding='utf-8').splitlines() if (root/'checkpoints/pretrain/metrics.jsonl').exists() else []:
        try:
            duration=json.loads(line).get('update_seconds')
        except json.JSONDecodeError:
            continue
        if isinstance(duration,(int,float)) and math.isfinite(duration) and duration>0:
            durations.append(duration)
    started_at=read_json(root/'status.json').get('started_at',time.time())
    logger.info('恢复监控游标：positions=%s，steps=%s；训练进程继续独立运行',positions,steps)
    previous_state=None
    final_state='success'
    try:
        while True:
            for stage in positions:
                path=root/'checkpoints'/stage/'metrics.jsonl'
                if not path.exists():
                    continue
                lines=path.read_text(encoding='utf-8').splitlines()
                for index in range(positions[stage],len(lines)):
                    try:
                        row=json.loads(lines[index])
                    except json.JSONDecodeError:
                        break
                    data=map_training_record(stage,row)
                    if data:
                        data['监控/已运行分钟']=max(0,(time.time()-started_at)/60)
                        run.log(data)
                        steps[stage]=max(steps[stage],row['step'])
                        duration=row.get('update_seconds')
                        if stage=='pretrain' and isinstance(duration,(int,float)) and math.isfinite(duration) and duration>0:
                            durations.append(duration)
                    positions[stage]=index+1
            current=effective_status(read_json(root/'status.json'), read_json(root/'completion_receipt.json'))
            limits=read_json(root/'run_limits.json')
            scalars={'监控/已运行分钟':max(0,(time.time()-started_at)/60),
                     '进度/预训练完成百分比':steps['pretrain']/TOTAL_PRETRAIN_STEPS*100,
                     '进度/已训练token数':steps['pretrain']*TOKENS_PER_UPDATE,
                     '进度/SFT完成百分比':steps['sft']/TOTAL_SFT_STEPS*100,
                     '时间/距预算停止小时':max(0,(limits.get('soft_deadline',time.time())-time.time())/3600)}
            if durations and steps['pretrain']<TOTAL_PRETRAIN_STEPS:
                # 给验证、保存和备份增加 15% 余量，再预留 SFT；实际完成时刻仍可能变化。
                remaining=(TOTAL_PRETRAIN_STEPS-steps['pretrain'])*statistics.median(durations[-20:])*1.15+1200
                scalars['时间/预计剩余小时']=remaining/3600
            if current.get('completed_pretrain_updates'):
                scalars['备份/已持久化预训练step']=current['completed_pretrain_updates'] if current.get('status')=='segment_complete' else read_json(persistent/'pretrain/LATEST.json').get('step',0)
            budget=read_json(root/'budget_usage.json')
            if budget:
                scalars.update({'成本/已用CU_账单':budget['used_cu'],'成本/剩余额度CU':budget['balance_cu'],
                                '成本/账单读数延迟秒':max(0,time.time()-budget['time'])})
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
                util,used,total,temp=[float(x.strip()) for x in result.stdout.strip().split(',')]
                scalars.update({'资源/GPU利用率百分比':util,'资源/显存占用GiB':used/1024,'资源/显存容量GiB':total/1024,'资源/GPU温度':temp})
            except (subprocess.SubprocessError,ValueError):
                logger.warning('GPU 资源采样暂时失败，继续同步训练指标')
            run.log(scalars)
            state_label=(current.get('status'),current.get('stage'),current.get('persistent_checkpoint'))
            if state_label!=previous_state:
                descriptions={'running':'训练进行中','backing_up':'正在备份模型到 Drive','segment_complete':'阶段断点已保存',
                              'stopped_budget':'预算保护已停止训练','failed':'训练异常，需要查看日志',
                              'completed_needs_manual_review':'训练与生成已完成，等待人工评分',
                              'completed_saved_for_manual_test':'训练完成，检查点已保存；问答测试待用户明天进行'}
                text=descriptions.get(current.get('status'),'启动准备中')+'\n当前阶段：'+str(current.get('stage','准备'))
                if current.get('persistent_checkpoint'):text+='\n最新完整备份：'+current['persistent_checkpoint']
                if current.get('error'):text+='\n错误：'+current['error']
                run.log_text(key='运行状态/说明',data=text)
                logger.info('监控状态更新：%s',text.replace('\n','；'))
                previous_state=state_label
            (root/'monitor_heartbeat.json').write_text(json.dumps({'time':time.time(),'steps':steps,'positions':positions,
                'url':run.url,'pipeline_status':current.get('status'),'stage':current.get('stage'),
                'resources':{key:value for key,value in scalars.items() if key.startswith('资源/')},
                'estimated_remaining_hours':scalars.get('时间/预计剩余小时')}))
            if current.get('status') in ('completed_needs_manual_review','completed_saved_for_manual_test','stopped_budget','failed'):
                final_state='crashed' if current['status']=='failed' else 'success'
                break
            time.sleep(args.interval)
    except Exception:
        final_state='crashed'
        logger.exception('监控进程异常；训练进程独立运行，不会因此中止')
        raise
    finally:
        run.finish(state=final_state,error='监控或训练异常，查看运行日志' if final_state=='crashed' else None,async_log_timeout=60)
        (root/'monitor_finished.json').write_text(json.dumps({'time':time.time(),'state':final_state}))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
