from pathlib import Path
import csv
import json
import os
from xml.sax.saxutils import escape

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'


def load_csv(name):
    with (RESULTS / name).open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def find_font():
    candidates = []
    if os.environ.get('REPORT_FONT'):
        candidates.append(Path(os.environ['REPORT_FONT']))
    try:
        import matplotlib
        candidates.append(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf')
    except ImportError:
        pass
    candidates.extend([Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                       Path('/Library/Fonts/DejaVuSans.ttf')])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('Set REPORT_FONT to DejaVuSans.ttf or install matplotlib.')


font_path = find_font()
pdfmetrics.registerFont(TTFont('DejaVu', str(font_path)))
bold_path = font_path.with_name('DejaVuSans-Bold.ttf')
pdfmetrics.registerFont(TTFont('DejaVuBold', str(bold_path if bold_path.exists() else font_path)))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVuBold')
styles = {
    'title': ParagraphStyle('Title', fontName='DejaVuBold', fontSize=20, leading=25, spaceAfter=13),
    'h1': ParagraphStyle('Heading', fontName='DejaVuBold', fontSize=15, leading=20, spaceAfter=10),
    'h2': ParagraphStyle('Subheading', fontName='DejaVuBold', fontSize=11, leading=15, spaceBefore=9, spaceAfter=6),
    'body': ParagraphStyle('Body', fontName='DejaVu', fontSize=9.5, leading=14, spaceAfter=8),
    'small': ParagraphStyle('Small', fontName='DejaVu', fontSize=8, leading=11.5, spaceAfter=6),
    'caption': ParagraphStyle('Caption', fontName='DejaVu', fontSize=8, leading=11, spaceAfter=9),
    'cell': ParagraphStyle('Cell', fontName='DejaVu', fontSize=8, leading=11),
    'center': ParagraphStyle('Center', fontName='DejaVu', fontSize=9, leading=13, alignment=TA_CENTER, spaceAfter=8),
}
summary = load_csv('summary.csv')
metrics = load_csv('metrics.csv')
gradients = load_csv('gradients.csv')
means = load_csv('mean_validation.csv')
data = json.loads((RESULTS / 'data_diagnostics.json').read_text())
final = json.loads((RESULTS / 'final_evaluation.json').read_text())
environment = json.loads((RESULTS / 'environment.json').read_text())
configurations = json.loads((RESULTS / 'configurations.json').read_text())


def value(config, seed, epoch, field='val_loss'):
    return float(next(r[field] for r in metrics if r['config'] == config and int(r['seed']) == seed and int(r['epoch']) == epoch))


def norm(config, seed, epoch, layer):
    return float(next(r['gradient_norm'] for r in gradients if r['config'] == config and int(r['seed']) == seed and int(r['epoch']) == epoch and int(r['layer']) == layer))


def crossing(config, seed, threshold=0.2):
    return min(int(r['epoch']) for r in metrics if r['config'] == config and int(r['seed']) == seed and float(r['val_loss']) < threshold)


story = []


def p(text, kind='body'):
    story.append(Paragraph(text, styles[kind]))


def table(headers, rows, widths):
    cells = [[Paragraph(escape(str(v)), styles['cell']) for v in row] for row in [headers] + rows]
    result = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef5')),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#d0d7de')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
    ]))
    story.append(result)
    story.append(Spacer(1, 9))


def chart(name, width, caption):
    from PIL import Image as PILImage
    with PILImage.open(FIGURES / name) as source:
        height = width * source.height / source.width
    story.append(Image(str(FIGURES / name), width=width, height=height))
    p(caption, 'caption')


def page(title):
    if story:
        story.append(PageBreak())
    p(title, 'h1')


# Отчёт собирается из результатов полного выполнения notebook.ipynb.
p('Диагностика обучения MLP', 'title')
p('Лабораторная работа № 1 по DLNN', 'h2')
p('Цель работы - обучить многослойный перцептрон для бинарной классификации и исследовать активации, инициализацию, глубину сети и заданные режимы оптимизации. Все выводы ниже опираются на 12 выполненных запусков на CPU.')
p(f'По заданному правилу выбрана конфигурация <b>{final["config"]}</b>, seed <b>{final["seed"]}</b>, эпоха 100. Итоговая test accuracy равна <b>{final["test_accuracy"]:.4f}</b>, test loss - <b>{final["test_loss"]:.6f}</b>.')
p('1. Подготовка данных', 'h1')
p('Сгенерированы 2400 объектов командой make_moons(n_samples=2400, noise=0.25, random_state=2026). Два последовательных стратифицированных train_test_split используют random_state=2026 и test_size=0.2, затем 0.25. Разбиение одинаково для всех экспериментов.')
table(['Выборка', 'Объекты', 'Класс 0', 'Класс 1'],
      [[name, data['sizes'][name], *data['class_counts'][name]] for name in ['train', 'validation', 'test']], [170, 100, 100, 100])
p(f'StandardScaler обучен только на train. Средние после преобразования: {data["train_mean"][0]:.3e} и {data["train_mean"][1]:.3e}; стандартные отклонения: {data["train_std"][0]:.6f} и {data["train_std"][1]:.6f}. Условия близости к 0 и 1 выполнены. Статистики validation и test не используются при обучении scaler, чтобы избежать утечки информации.')
chart('train_scatter.png', 420, 'Рисунок 1. Исходные обучающие объекты до масштабирования, цветом обозначен класс.')

page('2. Архитектура и протокол обучения')
p('MLP получает два признака, содержит L полносвязных скрытых слоёв ширины 32 и линейный выход из двух логитов. Во всех скрытых слоях одного запуска используется одна активация. Softmax перед CrossEntropyLoss не применяется.')
p('h₀ = x; zₗ = Wₗhₗ₋₁ + bₗ; hₗ = φ(zₗ), l = 1,…,L.<br/>z<sub>out</sub> = W<sub>L+1</sub>h<sub>L</sub> + b<sub>L+1</sub> ∈ R².', 'center')
p('Первый скрытый слой имеет W формы (32, 2), остальные - (32, 32), выходной - (2, 32). Размеры bias: 32 для скрытых слоёв и 2 для выхода.')
p('Число параметров', 'h2')
p('N(L) = (2·32 + 32) + (L − 1)·(32·32 + 32) + (32·2 + 2).<br/>N(1) = 162; N(6) = 5442. Оба значения совпали с sum(p.numel() for p in model.parameters()).')
table(['Код', 'L', 'Активация', 'Скрытые веса', 'Оптимизатор', 'LR'],
      [[c['name'], c['depth'], c['activation'], 'Xavier normal' if c['initialization']=='xavier' else 'He normal', c['optimizer'], c['lr']] for c in configurations], [32, 26, 90, 115, 110, 67])
p('Xavier normal задан с gain=1.0. He/Kaiming normal - с mode=fan_in, nonlinearity=relu. Выходной Linear во всех вариантах инициализирован Xavier normal с gain=1.0, все bias равны нулю. Для SGD установлен momentum=0; остальные параметры оптимизаторов, кроме LR, оставлены по умолчанию.')
p('Реализация и воспроизводимость', 'h2')
p('Использованы torch.nn, autograd, torch.optim.SGD/Adam и библиотечные матричные операции. Конструктор MLP, инициализация, цикл обучения, расчёт метрик, диагностика и запуск экспериментов реализованы самостоятельно. Готовые классификаторы и автоматические тренеры не используются.')
p('Каждый запуск: 100 эпох, batch size 128, перемешивание train в каждой эпохе. До создания модели вызывается torch.manual_seed(seed), для DataLoader создаётся новый torch.Generator().manual_seed(10000 + seed). Seed: 11 и 29. Устройство cpu, torch.set_num_threads(1), признаки float32, метки torch.long.')
p('Шаг обучения: zero_grad → forward → CrossEntropyLoss(mean) → backward → optimizer.step. На эпохе 0 и после каждой эпохи loss и accuracy оцениваются на полных train и validation в eval/no_grad; перед обучением включается train. Accuracy вычисляется через argmax как доля от 0 до 1. Проверяется конечность loss. Scheduler, dropout, batch normalization, weight decay, early stopping и предварительное обучение отсутствуют.')
p('Самопроверка', 'h2')
p('Для L=1 и L=6 проверены вход (8, 2), выход (8, 2), скалярный loss. После backward градиенты весов всех скрытых слоёв существуют, имеют формы соответствующих весов и конечные значения. Проверки пройдены.')

page('3. Итоговые метрики всех запусков')
p('Таблица 1. Метрики эпохи 100. Accuracy указана долей правильных ответов.', 'caption')
table(['Код', 'Seed', 'Train loss', 'Train acc.', 'Val loss', 'Val acc.'],
      [[r['config'], r['seed'], *[f'{float(r[k]):.6f}' for k in ['train_loss','train_accuracy','val_loss','val_accuracy']]] for r in summary], [35, 38, 99, 99, 99, 80])
p('Таблица 2. Средний validation loss эпохи 100 по двум seed.', 'caption')
table(['Конфигурация', 'A', 'B', 'C', 'D', 'E', 'F'],
      [['Mean val loss'] + [f'{float(r["mean_val_loss"]):.6f}' for r in means]], [110, 60, 60, 60, 60, 60, 60])
p('Активации при Xavier normal', 'h2')
p(f'На эпохе 1 при seed 11 validation loss A/B/C равен {value("A",11,1):.4f}/{value("B",11,1):.4f}/{value("C",11,1):.4f}. При seed 11 на эпохе 20 validation loss A/B/C равен {value("A",11,20):.4f}/{value("B",11,20):.4f}/{value("C",11,20):.4f}. Sigmoid обучается заметно медленнее; впервые val loss &lt; 0.2 достигается на эпохах {crossing("A",11)}/{crossing("B",11)}/{crossing("C",11)}. На эпохе 100 порядок меняется: C ({value("C",11,100):.4f}) лучше A ({value("A",11,100):.4f}) и B ({value("B",11,100):.4f}).')
p(f'При seed 29 раннее отставание A повторяется: loss эпохи 20 A/B/C = {value("A",29,20):.4f}/{value("B",29,20):.4f}/{value("C",29,20):.4f}; порог 0.2 достигнут на эпохах {crossing("A",29)}/{crossing("B",29)}/{crossing("C",29)}. Однако на эпохе 100 A ({value("A",29,100):.4f}) лучше B ({value("B",29,100):.4f}) и C ({value("C",29,100):.4f}). Раннее замедление Sigmoid устойчиво в этих двух запусках, итоговое превосходство ReLU - нет. Вывод относится к Xavier normal с gain=1.0.')
p('Инициализация ReLU', 'h2')
p(f'Дисперсия He равна 2/fan_in, Xavier - 2/(fan_in + fan_out): в квадратных слоях это 2/32 против 1/32, в первом - 1 против 2/34. Масштаб меняет активации и градиенты. На эпохе 20 переход C → D снижает validation loss с {value("C",11,20):.4f} до {value("D",11,20):.4f} при seed 11 и с {value("C",29,20):.4f} до {value("D",29,20):.4f} при seed 29. Но на эпохе 100 D хуже C при seed 11 и лучше при seed 29. Средний итоговый loss D ({float(means[3]["mean_val_loss"]):.4f}) выше C ({float(means[2]["mean_val_loss"]):.4f}); He не гарантирует лучшую validation-оценку.')

page('4. Динамика обучения и сравнения режимов')
chart('validation_loss.png', 415, 'Рисунок 2. Validation loss A-F по эпохам, seed 11.')
chart('validation_accuracy.png', 415, 'Рисунок 3. Validation accuracy A-F по эпохам, seed 11.')
p(f'<b>D/E.</b> Порог val loss &lt; 0.2 достигнут D на эпохах {crossing("D",11)}/{crossing("D",29)}, E - на {crossing("E",11)}/{crossing("E",29)} (seed 11/29). На эпохе 100 D лучше E для обоих seed. В E между эпохами 99 и 100 train loss растёт {value("E",11,99,"train_loss"):.4f} → {value("E",11,100,"train_loss"):.4f} и {value("E",29,99,"train_loss"):.4f} → {value("E",29,100,"train_loss"):.4f}; одновременно растёт validation loss. Это признак нестабильной оптимизации, а не только переобучения. Сравнение относится к Adam 0.003 и SGD 0.05: влияние оптимизатора отдельно от LR не установлено.', 'small')
p(f'<b>D/F.</b> D быстрее достигает порога 0.2 (6/5 против 35/42 эпох), но F имеет меньший итоговый validation loss при обоих seed. D лучше подгоняет train: loss {value("D",29,100,"train_loss"):.4f} против {value("F",29,100,"train_loss"):.4f} у F для seed 29. В D 5442 параметра против 162 у F; влияние глубины нельзя отделить от изменения числа параметров.', 'small')

page('5. Диагностика градиентов по слоям')
p('Контрольный batch - первые 128 объектов train после разбиения и масштабирования в сохранённом порядке. На эпохах 0, 20 и 100 отдельно выполняются zero_grad, forward, backward, измерение нормы и повторный zero_grad; параметры не обновляются. Измеряется gₗ(t) = ||∂CE(fθ(t)(Xprobe), yprobe)/∂Wₗ||F.', 'small')
for index, name in enumerate(['A', 'C', 'D'], start=4):
    chart(f'gradients_{name}.png', 325, f'Рисунок {index}. Нормы градиентов скрытых весов {name}, seed 11; логарифмическая ось Y.')

page('6. Объяснение градиентной диагностики')
p('Таблица 3. Нормы первого и последнего скрытых слоёв. Полные значения всех шести слоёв и обоих seed сохранены в results/gradients.csv.', 'caption')
grad_rows=[]
for name in ['A','C','D']:
    for seed in [11,29]:
        for epoch in [0,20,100]:
            grad_rows.append([name, seed, epoch, f'{norm(name,seed,epoch,1):.3e}', f'{norm(name,seed,epoch,6):.3e}'])
table(['Код', 'Seed', 'Эпоха', 'g₁', 'g₆'], grad_rows, [50,50,70,150,150])
p(f'<b>Sigmoid и правило цепочки.</b> При seed 11 начальные g₁/g₆ в A равны {norm("A",11,0,1):.3e}/{norm("A",11,0,6):.3e}: ранний слой имеет примерно в {norm("A",11,0,6)/norm("A",11,0,1):.0f} раза меньшую норму. При seed 29 разница составляет {norm("A",29,0,6)/norm("A",29,0,1):.0f} раза. В цепочке производных Sigmoid множители φ′(z) ≤ 1/4 могут многократно ослаблять сигнал. В C начальные нормы первого и шестого слоёв одного порядка: {norm("C",11,0,1):.3f}/{norm("C",11,0,6):.3f} и {norm("C",29,0,1):.3f}/{norm("C",29,0,6):.3f}. У активных ReLU производная равна 1; однако веса и неактивные нейроны также влияют на результат.', 'small')
p(f'<b>C/D.</b> На эпохе 0 He повышает g₁ с {norm("C",11,0,1):.3f} до {norm("D",11,0,1):.3f} (seed 11) и с {norm("C",29,0,1):.3f} до {norm("D",29,0,1):.3f} (seed 29); масштаб g₆ также выше. На эпохе 100 в D g₁ ниже C, а g₆ выше C для обоих seed. Стартовые масштабы и дальнейшая динамика различаются; большее начальное значение нормы не означает меньший итоговый validation loss.', 'small')
p('Все 108 норм конечны и положительны; нулевые нормы отсутствуют. Основные наблюдения повторяются на seed 29: ослабление раннего градиента A в начале и увеличение начальных норм D относительно C. Конкретные величины после обучения сильно различаются.', 'small')

page('7. Выбор модели и оценка обобщения')
p('По каждой конфигурации усреднён validation loss эпохи 100 для seed 11 и 29. Выбрана минимальная пара (средний loss, буква); затем внутри конфигурации - минимальная пара (loss, seed). Это реализует разрешение точных равенств по более ранней букве и seed 11. Промежуточные эпохи не участвуют в выборе.')
p(f'Минимум среднего validation loss: F, {final["mean_val_loss"]:.6f}. Для F loss seed 11 равен {value("F",11,100):.6f}, seed 29 - {value("F",29,100):.6f}; выбран seed 29. Загружено сохранённое состояние эпохи 100. Только после выбора выполнена оценка test; её результаты не использовались для изменения модели.')
table(['Выборка', 'Cross-entropy loss', 'Accuracy'],
      [['Train', f'{final["train_loss"]:.6f}', f'{final["train_accuracy"]:.6f}'],
       ['Validation', f'{final["val_loss"]:.6f}', f'{final["val_accuracy"]:.6f}'],
       ['Test', f'{final["test_loss"]:.6f}', f'{final["test_accuracy"]:.6f}']], [150,160,160])
chart('selected_loss.png', 450, 'Рисунок 7. Train и validation loss выбранного запуска F, seed 29.')
p(f'На train модель правильно классифицирует {final["train_accuracy"]:.2%} объектов; loss снижается с {value("F",29,0,"train_loss"):.4f} до {final["train_loss"]:.4f}. Обучение не осталось на плато случайного угадывания. Validation loss снижается с {value("F",29,0):.4f} до {final["val_loss"]:.4f}; разрыв с train на последней эпохе равен {final["val_loss"]-final["train_loss"]:.4f}.')
p('Небольшой положительный разрыв train/validation у F указывает на различие подгонки и обобщения, но не на сильное ухудшение validation на протяжении всего обучения. Test accuracy около 0.95 согласуется с хорошим качеством на отдельной отложенной выборке. Test loss ниже validation loss может объясняться различием конкретных объектов и шума; это не доказывает систематически лучшее качество на test. Оценка относится к одному фиксированному разбиению.')
p('В более глубокой D train loss ниже, а validation loss после раннего улучшения может расти: это согласуется с переобучением. В E скачок одновременно на train и validation указывает на проблему оптимизации при заданном режиме. Эти механизмы следует различать.')

page('8. Выводы и воспроизводимость')
p('Почему нормы недостаточно для оценки скорости обучения', 'h2')
p('Градиент весов слоя имеет вид усреднённого внешнего произведения обратного сигнала δₗ и входной активации hₗ₋₁. Поэтому его норма зависит и от ошибки, и от масштаба входов. В первой матрице 32·2 = 64 элемента, в последующих скрытых матрицах 32·32 = 1024; нормы Фробениуса разных слоёв не нормированы по числу элементов. Нельзя напрямую интерпретировать их отношение как чистую меру затухания обратного сигнала.')
p('Изменение loss зависит от направления шага, LR, геометрии функции и оптимизатора. Adam преобразует градиенты с учётом накопленных моментов, поэтому норма исходного градиента не равна норме обновления. Диагностика проводится на одном train-batch, тогда как validation loss измеряет качество на других объектах. Большая норма не гарантирует быстрое снижение validation loss.')
p('Три вывода', 'h2')
p(f'<b>1. Раннее обучение глубокой Sigmoid затруднено.</b> При обоих seed начальный g₁ в A в {norm("A",11,0,6)/norm("A",11,0,1):.0f}/{norm("A",29,0,6)/norm("A",29,0,1):.0f} раза меньше g₆; порог val loss 0.2 достигается на эпохах 66/56 против 4/4 у C. Это согласуется с ослаблением произведения производных Sigmoid по правилу цепочки. К эпохе 100 A всё же достигает хорошего качества, и универсального итогового превосходства ReLU наблюдения не подтверждают.')
p(f'<b>2. Масштаб инициализации влияет на градиенты, но не гарантирует обобщение.</b> В ReLU переход Xavier → He повышает стартовый g₁ с {norm("C",11,0,1):.3f} до {norm("D",11,0,1):.3f} для seed 11 и ускоряет часть ранней динамики к эпохе 20. При этом средний итоговый validation loss D ({float(means[3]["mean_val_loss"]):.4f}) выше C ({float(means[2]["mean_val_loss"]):.4f}). Норма градиента не является самостоятельной мерой качества.')
p(f'<b>3. Более сильная подгонка train не обеспечивает лучшее качество validation.</b> У D при seed 29 train loss {value("D",29,100,"train_loss"):.4f}, у F - {value("F",29,100,"train_loss"):.4f}; validation loss соответственно {value("D",29,100):.4f} и {value("F",29,100):.4f}. Выбранная F с 162 параметрами имеет меньший средний validation loss, чем D с 5442 параметрами. Результат согласуется с различием ёмкости моделей и обобщения, но не изолирует эффект глубины.')
p('Ограничения', 'h2')
p('Недостаточно данных для утверждения «Adam всегда лучше SGD», «He всегда лучше Xavier» или «мелкие сети всегда обобщают лучше глубоких». В D/E одновременно меняются оптимизатор и LR, в D/F - глубина и число параметров. A-C исследованы только с Xavier gain=1.0. Два seed и одно разбиение не дают надёжной оценки распределения качества или статистической значимости; LR фиксированы и не подбирались.')
p('Материалы и запуск', 'h2')
p('notebook.ipynb последовательно воспроизводит все 12 запусков, 1212 строк метрик эпох 0-100, 108 строк градиентов, сравнительные таблицы, семь графиков, выбор и test. results/ содержит CSV и JSON, figures/ - PNG. README.md описывает установку и запуск; requirements.txt фиксирует версии. Отчёт собирается scripts/build_report.py из сохранённых результатов. GitHub-репозиторий используется вместо ZIP по выбранному формату сдачи.', 'small')
p(f'Фактическое окружение: Python {environment["python"]}; CPU, один поток PyTorch. NumPy {environment["versions"]["numpy"]}; scikit-learn {environment["versions"]["scikit-learn"]}; PyTorch {environment["versions"]["torch"]}; Matplotlib {environment["versions"]["matplotlib"]}.', 'small')
p('Источник условий: lab1.docx, «Лабораторная работа. Диагностика обучения MLP». Данные: sklearn.datasets.make_moons. Все численные результаты получены собственным выполнением ноутбука.', 'small')


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont('DejaVu', 8)
    canvas.setFillColor(colors.HexColor('#64748b'))
    canvas.drawString(52, 25, 'DLNN · Лабораторная работа 1')
    canvas.drawRightString(A4[0] - 52, 25, str(document.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(ROOT / 'report.pdf'), pagesize=A4,
                        leftMargin=52, rightMargin=52, topMargin=42, bottomMargin=44,
                        title='Диагностика обучения MLP', author='',
                        subject='Лабораторная работа 1 по DLNN')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(ROOT / 'report.pdf')
