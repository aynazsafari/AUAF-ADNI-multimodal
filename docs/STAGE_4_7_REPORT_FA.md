# گزارش نهایی مرحله ۴.۷ — اعتبارسنجی آماری و ممیزی چندمتغیره Circularity

## وضعیت اجرا

- تحلیل روی خروجی‌های نهایی Stage 4.5 و داده‌های as-of-index مرحله 4.4 اجرا شد؛ هیچ split جدیدی ساخته نشد.
- ۱٬۰۵۰ فایل پیش‌بینی فردبه‌فرد (۵۲۵ full-test و ۵۲۵ matched) بررسی شد؛ همه rosterها، شناسه‌ها، برچسب‌ها و احتمال‌ها از ممیزی یکپارچگی عبور کردند.
- ۱٬۰۰۰ تکرار bootstrap طبقه‌بندی‌شده بر حسب کلاس برای فاصله اطمینان و مقایسه‌های paired استفاده شد.
- پنج seed، پنج initialization روی همان split ثابت هستند؛ پنج fold یا پنج cohort مستقل نیستند.
- ۱۵ مدل AUAF و ۱۵ مدل baseline در آزمایش حساسیت circularity، با همان train/validation/testهای منجمد، دوباره آموزش داده شدند.

## ۱. بازتولید معیارهای Stage 4.5

- معیارهای accuracy، F1، Brier، log-loss و ECE در تمام ۱٬۰۵۰ شرط با خروجی آرشیوی عملاً دقیق بازتولید شدند.
- برای ROC-AUC/PR-AUC اختلاف کوچک عددی در خروجی ExtraTrees مشاهده شد؛ بیشترین اختلاف در matched diagnosis multiclass برای یک شرط seed/missingness برابر 0.000306 در ROC-AUC و 0.000101 در PR-AUC بود. این اختلاف در گزارش round-trip جداگانه علامت‌گذاری شده و پنهان نشده است.

## ۲. Bootstrap و مقایسه مدل‌ها

- برای هر مدل، task، cohort و پنج سطح missingness، میانگین و SD پنج seed به‌همراه ۹۵٪ bootstrap CI محاسبه شد.
- در مقایسه AUAF با AttentionFusion و سه baseline درختی، ۱٬۴۴۰ مقایسه paired ثبت شد. ۱۳۷ ردیف پس از BH-FDR در کل خانواده مقایسه‌ها معنادار شدند؛ این شمار به معنی برتری کلی AUAF نیست و باید در سطح task/metric/شرط خوانده شود.
- در سطح ۰٪ حذف اضافه biomarker، برای چهار معیار کلیدی F1، ROC-AUC، Brier و ECE هیچ مقایسه AUAF در برابر comparator پس از FDR معنادار نشد.
- در تحلیل اثر ۲۵/۵۰/۷۵/۱۰۰٪ حذف اضافه biomarker نسبت به ۰٪، هیچ‌یک از ۲٬۰۱۶ اثر پس از BH-FDR معنادار نماند. این نتیجه نبود اثر قطعی را اثبات نمی‌کند؛ matched cohortها کوچک‌اند، به‌ویژه conversion با n=29.

### چند مقدار نمونه در full test، سطح ۰٪

| Task | مدل | Macro-F1 | ROC-AUC |
|---|---|---:|---:|
| diagnosis_multiclass | AUAF | 0.849 | 0.956 |
| diagnosis_multiclass | XGBoost (best tree by validation) | 0.840 | 0.960 |
| diagnosis_binary | AUAF | 0.797 | 0.930 |
| diagnosis_binary | CatBoost (best tree by validation) | 0.816 | 0.936 |
| mci_conversion | AUAF | 0.721 | 0.799 |
| mci_conversion | CatBoost (best tree by validation) | 0.737 | 0.829 |

## ۳. Temperature scaling و thresholding

- دمای calibration برای هر task/model/seed فقط روی validation fit شد؛ همان دما برای test به کار رفت.
- pre/post calibration در دو نقطه ۰٪ و ۱۰۰٪ missingness گزارش شده است. هیچ اثر calibration پس از BH-FDR در ۱٬۰۰۸ ردیف مقایسه باقی نماند.
- برای diagnosis_binary و mci_conversion، threshold با Youden J فقط روی validation انتخاب و برای test در پنج سطح missingness اعمال شد.

## ۴. ممیزی چندمتغیره diagnostic circularity

- بر اساس feature manifest و reconciliation واقعی Stage 4.4، ۲۰ متغیر از بلوک‌های CDR، FAQ، MMSE و MoCA پیش از preprocessing حذف شدند.
- فهرست حذف‌شده: CDMEMORY, CDRSB, CLOCKCON, CLOCKHAN, CLOCKNO, FAQBEVG, FAQEVENT, FAQFINAN, FAQFORM, FAQGAME, FAQMEAL, FAQREM, FAQSHOP, FAQTOTAL, FAQTRAVL, FAQTV, FFLUENCY, MMSEASON, MOCA, TRAILS.
- مجموعه ورودی strict sensitivity از ۳۲ ویژگی مؤثر به ۱۲ ویژگی رسید: ۹ ویژگی بالینی/شناختی غیر از بلوک‌های فوق و ۳ CSF biomarker.
- baseline برتر فقط از بین سه baseline موجود در بسته Stage 4.5 (XGBoost، CatBoost، ExtraTrees) و با میانگین Validation Macro-F1 انتخاب شد: multiclass = XGBoost، binary = CatBoost، conversion = CatBoost.
- در full test، حذف بلوک‌های تشخیصی باعث افت محسوس عملکرد شد؛ این نتیجه نشان می‌دهد بخشی از قدرت پیش‌بینی به مقیاس‌های بالینی نزدیک به تعریف تشخیص وابسته است. این آزمایش گواه نبود leakage نیست و به‌عنوان sensitivity analysis گزارش می‌شود.

### Macro-F1 استاندارد در برابر strict sensitivity (میانگین پنج seed، full test)

| Task | مدل | همه ویژگی‌ها | پس از حذف ۲۰ ویژگی |
|---|---|---:|---:|
| diagnosis_multiclass | AUAF | 0.848 | 0.468 |
| diagnosis_binary | AUAF | 0.796 | 0.612 |
| mci_conversion | AUAF | 0.721 | 0.607 |
| diagnosis_multiclass | XGBoost | 0.839 | 0.435 |
| diagnosis_binary | CatBoost | 0.814 | 0.560 |
| mci_conversion | CatBoost | 0.737 | 0.599 |

- برای matched test، مقایسه روی همان افراد مشترک انجام شد: n=81 برای multiclass، n=67 برای binary و n=29 برای conversion. در conversion و زیرگروه‌های شدت، CIها عریض و برآوردها ناپایدارند.
- تحلیل severity با CDRSB و FAQTOTAL از median محاسبه‌شده روی train استفاده کرد. در matched conversion، CDRSB برای ۲۸ از ۲۹ نفر missing بود؛ بنابراین این strata قابل تفسیر قوی نیست. FAQTOTAL نیز گروه‌های بسیار کوچک دارد.

## ۵. Conformal prediction — ممیزی قابلیت بازتولید

- مقادیر Table 6 از Manuscript استخراج و از نظر سازگاری داخلی بررسی شدند؛ اما prediction setهای فردبه‌فرد، calibration scores/q-hatها و پیاده‌سازی دقیق RAPS-lite در آرشیو Stage 4.4/4.5 موجود نبود. بنابراین پوشش و اندازه setها در این مرحله از نو محاسبه نشدند.
- مقدار گزارش‌شده coverage: CN/MCI/AD = 0.9259، AD/Non-AD = 0.8851 (پایین‌تر از هدف 0.90)، conversion = 0.9705.
- برای multiclass، abstention_rate گزارش‌شده 0 است، درحالی‌که مکمل singleton_rate برابر 0.4211 است؛ تعریف abstention در manuscript باید روشن شود. این اختلاف به‌عنوان ابهام تعریف ثبت شد، نه اینکه خودسرانه اصلاح شود.

## ۶. Provenance و محدودیت‌ها

- source table، instrument، شرح ویژگی، missing-code handling و temporal alignment از `feature_manifest_34.csv` مرحله 4.4 ثبت شد.
- برای ویژگی‌های بالینی، انتخاب روی index visit و برای CSF biomarkers تطبیق PHASE + RID + VISCODE2 همراه شرط EXAMDATE_BIO ≤ INDEX_DATE ثبت شده است. در temporal audit، ۲۰۹ رکورد biomarker پس از index در cohort تشخیصی و ۱۰۳ رکورد در conversion شناسایی شدند و از ورودی as-of کنار گذاشته شدند.
- در audit زمانی، biomarker قابل استفاده در index برای ۳۱۶/۳۷۰۰ نفر در cohort تشخیصی و ۱۶۹/۹۱۱ نفر در conversion ثبت شده است؛ توزیع lag همه رکوردهای تاریخ‌دار به تفکیک در جدول temporal lag آمده است.
- شناسه ردیف خام/visit code اختصاصی هر feature در integrated predictor matrix نگهداری نشده؛ این محدودیت در inventory صریحاً علامت‌گذاری شده است.
- نسخه نهایی Stage 4.5 فقط دو modality دارد: clinical/cognitive و fluid biomarkers؛ هیچ modality تصویربرداری در ورودی مدل نهایی وجود ندارد.
- مطالعه همچنان ADNI-only و بدون external validation است. از واژه leakage-free یا ادعای آمادگی بالینی استفاده نشود.

## فایل‌های کلیدی

- `reports/AUAF_Stage4_7_Master_Results.xlsx`: همه جدول‌های اصلی در یک workbook.
- `tables/`: جدول‌های کامل seed-level، bootstrap CI، paired tests، missingness effects، calibration، thresholds و circularity.
- `predictions/`: calibrated predictions و reduced-feature predictions در CSV.GZ.
- `models/`: best checkpoints مدل‌های حساسیت AUAF و مدل‌های درختی reduced؛ last.pt در ZIP نهایی نگهداری نمی‌شود.
- `manifests/`: hashها، roster audit، feature provenance، فهرست حذف/نگهداشت و run manifest.

## جمع‌بندی

مرحله ۴.۷ از نظر تحلیل آماری Stage 4.5، calibration، thresholding، feature provenance و بازآموزی حساسیت چندمتغیره اجرا شد. تنها بخش conformal در سطح participant قابل بازتولید نبود، چون prediction setها و q-hatهای اصلی در ورودی‌های در دسترس وجود نداشتند؛ این مورد به‌عنوان gap صریح تحویل شده است.