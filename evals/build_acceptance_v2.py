"""冻结第二版 240 题；模板/情境不与程序训练模板共用，双语配对按组计数。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    rows = []

    def add(task, group, language, prompt, reference, scoring='exact', messages=None):
        row = {'id': f'v2_{task}_{group}_{language}', 'group': f'{task}/{group}', 'category': task,
               'language': language, 'prompt': prompt, 'reference': reference, 'scoring': scoring}
        if messages:
            row['messages'] = messages
        rows.append(row)

    # 小范围稳定常识，按语义评分，不能用关键词出现冒充事实正确。
    facts = [
        ('磁铁通常能吸住铁还是木头？', 'Does a magnet usually attract iron or wood?', '铁', 'Iron'),
        ('蝴蝶一生中会经历哪种幼虫阶段？', 'What larval stage does a butterfly go through?', '毛毛虫', 'Caterpillar'),
        ('哪种工具用于测量温度？', 'Which tool measures temperature?', '温度计', 'Thermometer'),
        ('通常用什么单位表示电流？', 'What is the usual unit of electric current?', '安培', 'Ampere'),
        ('一打物品是多少件？', 'How many items are in a dozen?', '十二件', 'Twelve'),
        ('铁轨接缝处预留空隙主要是为了应对什么变化？', 'Why are gaps left at railway rail joints?', '应对温度变化造成的热胀冷缩', 'To allow expansion and contraction as temperature changes'),
        ('鲸主要用鳃还是肺呼吸？', 'Do whales mainly breathe with gills or lungs?', '肺', 'Lungs'),
        ('声音在真空中能传播吗？', 'Can sound travel through a vacuum?', '不能，声音需要介质', 'No, sound needs a medium'),
        ('昼夜变化主要由地球的什么运动造成？', 'Which movement of Earth mainly causes day and night?', '自转', 'Rotation'),
        ('纸通常主要由哪类材料制成？', 'What material is paper usually mainly made from?', '植物纤维，如木浆', 'Plant fibers, such as wood pulp'),
        ('血液主要由哪个器官推动循环？', 'Which organ mainly pumps blood around the body?', '心脏', 'Heart'),
        ('钟表上分针走一整圈通常经过多久？', 'How long does one full turn of a clock minute hand usually take?', '一小时', 'One hour'),
        ('冰融化后通常变成什么？', 'What does ice normally become when it melts?', '液态水', 'Liquid water'),
        ('正方形有多少条等长边？', 'How many equal sides does a square have?', '四条', 'Four'),
        ('地球大气中含量最多的气体是什么？', 'What is the most abundant gas in Earth’s atmosphere?', '氮气', 'Nitrogen'),
        ('普通透明玻璃和木板，哪一个更容易让光透过？', 'Which transmits light more easily, ordinary clear glass or a wooden board?', '普通透明玻璃', 'Ordinary clear glass'),
        ('雨后彩虹与阳光通过什么有关？', 'A rainbow after rain involves sunlight passing through what?', '空气中的水滴', 'Water droplets in the air'),
        ('电脑断电后通常会失去 RAM 中的数据吗？', 'Does a computer normally lose data in RAM when power is removed?', '会，普通 RAM 是易失性内存', 'Yes, ordinary RAM is volatile'),
        ('把可回收物分类投放有什么作用？', 'Why separate recyclable waste?', '方便回收利用、减少资源浪费', 'It helps recycling and reduces resource waste'),
        ('借来的书到期前无法归还，合理的处理是什么？', 'What is a reasonable action if you cannot return a borrowed book by its due date?', '联系出借方申请续借或说明情况', 'Contact the lender to request renewal or explain the situation'),
    ]
    for i, (zh, en, azh, aen) in enumerate(facts):
        add('facts', i, 'zh', zh, azh, 'semantic')
        add('facts', i, 'en', en, aen, 'semantic')
    translations = [
        ('请把空杯子放回架子。', 'Please put the empty cup back on the shelf.'),
        ('会议结束后，我会给你回电话。', 'I will call you back after the meeting.'),
        ('这双鞋比那双轻。', 'This pair of shoes is lighter than that one.'),
        ('我们沿着河岸走了半小时。', 'We walked along the riverbank for half an hour.'),
        ('门没有锁，你可以进去。', 'The door is unlocked. You can go in.'),
        ('请在离开之前关掉台灯。', 'Please turn off the desk lamp before you leave.'),
        ('我把钥匙忘在办公室了。', 'I left my keys at the office.'),
        ('这辆公交车经过博物馆吗？', 'Does this bus pass the museum?'),
        ('如果下雨，我们就留在室内。', 'If it rains, we will stay indoors.'),
        ('这本书可以借两个星期。', 'This book can be borrowed for two weeks.'),
        ('别急，我们还有足够的时间。', 'Take your time. We still have enough time.'),
        ('她把围巾留在了椅子上。', 'She left her scarf on the chair.'),
        ('请告诉我最近的出口在哪里。', 'Please tell me where the nearest exit is.'),
        ('我想换一个靠窗的座位。', 'I would like to switch to a window seat.'),
        ('这张照片是去年秋天拍的。', 'This photo was taken last autumn.'),
        ('水壶里还有一点温水。', 'There is still a little warm water in the kettle.'),
        ('修理之前，请先拔掉电源插头。', 'Please unplug it before repairing it.'),
        ('他们正在为周末的旅行打包。', 'They are packing for the weekend trip.'),
        ('我听见有人在敲后门。', 'I heard someone knocking at the back door.'),
        ('这封信应该寄给谁？', 'Who should this letter be sent to?'),
    ]
    for i, (zh, en) in enumerate(translations):
        add('translation', i, 'zh', f'用英语表达这句话，不加解释：“{zh}”', en, 'semantic')
        add('translation', i, 'en', f'Render this sentence in Chinese, without commentary: "{en}"', zh, 'semantic')
    # 以下四类每类 20 个情境、40 个双语变体。任务家族有关联，不视为 160 个独立统计样本。
    for i in range(20):
        a, b, c = 13 + i * 3, 7 + i, 4 + i % 7
        number = a + b - c
        add('arithmetic', i, 'zh', f'仓库上午收到 {a} 箱，下午收到 {b} 箱，随后运走 {c} 箱。这批货现在剩几箱？只写数字。', str(number))
        add('arithmetic', i, 'en', f'A warehouse received {a} boxes in the morning and {b} in the afternoon, then shipped {c}. How many of these boxes remain? Digits only.', str(number))
        code = f'Q{i+31}'
        add('format', i, 'zh', f'输出一个 JSON 对象，只有键 "code" 和 "ready"。code 的值是 "{code}"，ready 的值是布尔值 true。不要代码块。', json.dumps({'code': code, 'ready': True}), 'json')
        add('format', i, 'en', f'Output a JSON object with only "code" and "ready". Set code to "{code}" and ready to the boolean true. No code block.', json.dumps({'code': code, 'ready': True}), 'json')
        early, late = 9 + i % 4, 15 + i % 4
        add('reading', i, 'zh', f'公告：青石馆原定 {early}:00 开门，因维修改为 {late}:00 开门。以新时间为准。\n现在几点开门？只写 HH:MM。', f'{late:02}:00')
        add('reading', i, 'en', f'Notice: Cedar Hall was due to open at {early}:00, but repairs moved opening to {late}:00. The new time applies.\nWhen does it open now? Reply only with HH:MM.', f'{late:02}:00')
        old, new = 200 + i, 300 + i
        for language in ('zh', 'en'):
            if language == 'zh':
                history = [('user', f'请记住，我预订的是 {old} 号房间。'), ('assistant', f'好的，你预订的是 {old} 号房间。'),
                           ('user', f'前台改成了 {new} 号房间，旧号码作废。'), ('assistant', f'已更新为 {new} 号房间。')]
                question = '我应该去哪个房间？只写号码。'
            else:
                history = [('user', f'Remember that I booked room {old}.'), ('assistant', f'Okay, you booked room {old}.'),
                           ('user', f'Reception changed it to room {new}; the old number no longer applies.'), ('assistant', f'Updated to room {new}.')]
                question = 'Which room should I go to? Reply with only the number.'
            messages = [{'role': role, 'content': content} for role, content in history] + [{'role': 'user', 'content': question}]
            add('multi_turn', i, language, question, str(new), messages=messages)
    assert len(rows) == 240 and len({row['id'] for row in rows}) == 240
    return rows


if __name__ == '__main__':
    target = ROOT / 'bilingual_acceptance_v2_holdout.jsonl'
    content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in build())
    if target.exists() and target.read_text(encoding='utf-8') != content:
        raise FileExistsError('验收题已冻结；修改需要新版本，不能覆盖已用验收题')
    target.write_text(content, encoding='utf-8')
    (ROOT / 'bilingual_acceptance_v2_holdout.manifest.json').write_text(json.dumps({
        'rows': 240, 'semantic_groups': 120, 'languages': {'zh': 120, 'en': 120},
        'sha256': hashlib.sha256(content.encode('utf-8')).hexdigest(), 'status': 'frozen_not_yet_run',
        'limitations': '项目内自建验收，部分题共享模板，双语题配对；不是权威榜单或240个独立统计样本。只在最终候选锁定后使用。'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
