document.addEventListener('DOMContentLoaded', () => {
    // Upload Page Logic
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadForm = document.getElementById('uploadForm');
    const loader = document.getElementById('loader');
    const submitBtn = document.getElementById('submitBtn');

    if (uploadArea) {
        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // File selection
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });

        // Drag and Drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            uploadArea.classList.add('dragover');
        }

        function unhighlight(e) {
            uploadArea.classList.remove('dragover');
        }

        uploadArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFile(files[0]);
            
            // Update file input manually so form submission works
            fileInput.files = files;
        });

        function handleFile(file) {
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onloadend = () => {
                    imagePreview.src = reader.result;
                    imagePreview.classList.add('active');
                    document.querySelector('.upload-area__content').style.display = 'none';
                }
            }
        }

        // Form Submit
        if (uploadForm) {
            uploadForm.addEventListener('submit', () => {
                loader.style.display = 'block';
                submitBtn.disabled = true;
                submitBtn.innerText = 'Обработка...';
            });
        }
    }

    // Result Page Logic
    const urlParams = new URLSearchParams(window.location.search);
    const arg = urlParams.get('arg');
    
    if (arg) {
        const title = document.getElementById('title');
        const text = document.getElementById('text');
        const image = document.getElementById('image');

        // Add class to image div for background image (handled in CSS if needed, or we can set it here if we had the image)
        // Since we don't have the uploaded image on the result page (unless we pass it or store it), 
        // we might want to show a generic image or the one corresponding to the breed if available in static/img.
        // The original code added a class like ".beagle" to the image div.
        
        if (image) {
            image.classList.add(arg);
            // Try to set background image if it exists in static/img/
            // Note: This assumes images are named like the class. 
            // If not, the CSS classes from original code should handle it.
            // Let's keep the class adding logic.
        }

        switch (arg) {
            case "Beagle":
                console.log("Beagle");
                title.innerHTML = "Бигль";
                text.innerHTML = "Бигль (англ. beagle — гончая) — охотничья порода собак, выведенная в Великобритании. Среднего размера, внешне сходна с фоксхаундом, но мельче, с более короткими ногами и более длинными и мягкими ушами. Рост — от 33 до 40 см в холке, вес — между 9 и 14 кг.";
                break;
            case "Corgi":
                console.log("corgi");
                title.innerHTML = "Корги";
                text.innerHTML = "Зародилась порода в Уэльсе, где она стала одной из первых пастушьих собак. Вельш-корги получили распространение в XX веке. Основой для выведения породы, вероятно, послужили шведский вальхунд или исландская собака.";
                break;
            case "Blenheim_spaniel":
                console.log("Blenheim_spaniel");
                title.innerHTML = "Спаниэль";
                text.innerHTML = "Cпаниель – приветливый умница с проникновенным взглядом и длинной атласной шерсткой. Из этих энергичных, общительных симпатяг получаются исключительные приятели.";
                break;
            case "Dalmatian":
                console.log("dalmatian");
                title.innerHTML = "Далматин";
                text.innerHTML = "Далматин принадлежит к числу пород, которым «медийность» и сиюминутная популярность принесли больше вреда, чем пользы. Собаки с непростым характером и высокими потребностями к ежедневным упражнениям.";
                break;
            case "German_shepherd":
                console.log("german_shepherd");
                title.innerHTML = "Овчарка";
                text.innerHTML = "Овчарка – завсегдатай верхушек рейтингов самых умных, самых преданных, самых обучаемых питомцев. Главным призванием немцев остается охрана порядка.";
                break;
            case "Huskies":
                console.log("huskies");
                title.innerHTML = "Хаски";
                text.innerHTML = "Хаски — заводская специализированная порода собак, выведенная чукчами северо-восточной части Сибири. Используется не только как ездовая, но и как собака-компаньон.";
                break;
            case "Labrador":
                console.log("Labrador");
                title.innerHTML = "Лабрадор";
                text.innerHTML = "Лабрадор получил столь широкое распространение благодаря удивительно удачному сочетанию внешних данных и «рабочих» качеств. Они регулярно попадают на верхние строчки рейтингов «самых преданных».";
                break;
            case "Shar_pei":
                console.log("shar_pei");
                title.innerHTML = "Шарпей";
                text.innerHTML = "Шарпей из тех пород, которые невозможно не заметить. Необычная форма головы и, конечно, фирменные складки выделяют их среди сородичей.";
                break;
            default:
                title.innerHTML = arg;
                text.innerHTML = "Информация об этой породе пока отсутствует в нашей базе данных, но мы определили её как " + arg;
                break;
        }
    }
});
