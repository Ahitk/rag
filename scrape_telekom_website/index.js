const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const urls = require('./sitemap.json');

const OUTPUT_FOLDER = './output_folder';
const MAX_CONCURRENT_REQUESTS = 5;

/**
 * Sanitize a URL to create a valid filename.
 */
const sanitizeFilename = (url) => {
    const filename = url.replace(/^https?:\/\//, '').replace(/[^a-zA-Z0-9]/g, '_');
    console.log(`Sanitized filename: ${filename}`);
    return filename;
};

/**
 * Fetch and extract text content from a given URL, excluding elements with certain classes or styles.
 */
const fetchPageContent = async (url) => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    try {
        console.log(`Loading page: ${url}`);
        await page.goto(url, { waitUntil: 'networkidle2' });

        console.log(`Extracting content from: ${url}`);
        const content = await page.evaluate(() => {
            let extractedText = '';
            document.querySelectorAll('*').forEach(element => {
                const tagName = element.tagName.toLowerCase();

                // Exclude elements with inline styles, certain classes, or irrelevant tags
                if (
                    ['script', 'style', 'meta', 'link', 'noscript', 'source'].includes(tagName) ||
                    element.hasAttribute('style')
                ) {
                    return;
                }

                const text = element.textContent.trim();
                if (text) extractedText += text + '\n';
            });
            return extractedText;
        });

        console.log(`Extracted content for ${url}`);
        return content;
    } catch (error) {
        console.error(`Error fetching content from ${url}: ${error.message}`);
    } finally {
        await browser.close();
    }
};

/**
 * Save content to a file.
 */
const saveContentToFile = async (filePath, content) => {
    try {
        console.log(`Writing content to file: ${filePath}`);
        await fs.promises.writeFile(filePath, content);
    } catch (error) {
        console.error(`Error writing to file ${filePath}: ${error.message}`);
    }
};

/**
 * Scrape a single URL and save its content.
 */
const scrapeUrl = async (url) => {
    console.log(`Processing URL: ${url}`);
    const sanitizedFilename = `${sanitizeFilename(url)}.txt`;
    const filePath = path.join(OUTPUT_FOLDER, sanitizedFilename);

    console.log(`Saving content to: ${filePath}`);
    const content = await fetchPageContent(url);
    await saveContentToFile(filePath, content);
};

/**
 * Process URLs in batches with concurrency limits.
 */
const processUrls = async () => {
    if (!fs.existsSync(OUTPUT_FOLDER)) {
        console.log(`Creating output folder: ${OUTPUT_FOLDER}`);
        fs.mkdirSync(OUTPUT_FOLDER, { recursive: true });
    }

    for (let i = 0; i < urls.length; i += MAX_CONCURRENT_REQUESTS) {
        const batch = urls.slice(i, i + MAX_CONCURRENT_REQUESTS);
        console.log(`Processing batch: ${batch}`);
        await Promise.all(batch.map(url => scrapeUrl(url)));
    }

    console.log('All URLs processed successfully.');
};

// Start processing the URLs
processUrls();
