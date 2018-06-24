#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include "stuff.h"

//#define RECONNECT 1


CURL *curl = NULL;
struct curl_slist *headers = NULL;
struct curl_slist *host = NULL;

void gcurl_init() {
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    /* GitHub commits API v3 requires a User-Agent header */
    //headers = curl_slist_append(headers, "Host: 127.0.0.1");
    //headers = curl_slist_append(headers, "User-Agent: assetchain-switcher");

    //host = curl_slist_append(NULL, "localhost:7771:127.0.0.1");
    //host = curl_slist_append(NULL, "127.0.0.1:7771:127.0.0.1");

    //CURLM *m = curl_multi_init();
    //curl_multi_setopt(m, CURLMOPT_MAXCONNECTS, 1000L);
}

void gcurl_cleanup() {
    if (curl) curl_easy_cleanup(curl);
    if (headers) curl_slist_free_all(headers);
    if (host) curl_slist_free_all(host);
    curl_global_cleanup();

}

/* Notes:

 - https://curl.haxx.se/libcurl/c/post-callback.html -curl post example
 - static functions are functions that are only visible to other functions in the same file

 */

/* Return the offset of the first newline in text or the length of
   text if there's no newline */

/*
static int newline_offset(const char *text)
{
    const char *newline = strchr(text, '\n');
    if(!newline)
        return strlen(text);
    else
        return (int)(newline - text);
}
*/

struct write_result
{
    char *data;
    int pos;
};

static size_t write_response(void *ptr, size_t size, size_t nmemb, void *stream)
{
    struct write_result *result = (struct write_result *)stream;

    if(result->pos + size * nmemb >= BUFFER_SIZE - 1)
    {
        fprintf(stderr, "error: too small buffer\n");
        return 0;
    }

    memcpy(result->data + result->pos, ptr, size * nmemb);
    result->pos += size * nmemb;

    return size * nmemb;
}

char *daemon_request(char *daemon_ip, int daemon_port, char *daemon_username, char *daemon_password, char *command)
{

    char url[256];
    //char userpwd[256];
    snprintf(url, 256, "http://%s:%d", daemon_ip, daemon_port);
    //snprintf(userpwd, 256, "%s:%s", daemon_username, daemon_password);
    //printf("url: %s\n",url);


    CURLcode status;

    char *data = NULL;
    long code;

    #ifdef RECONNECT
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    #endif

    if(!curl)
        goto error;

    data = malloc(BUFFER_SIZE);
    if(!data)
        goto error;

    struct write_result write_result = {
        .data = data,
        .pos = 0
    };

    // https://curl.haxx.se/libcurl/c/curl_easy_setopt.html

    //curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);

    //curl_easy_setopt(curl, CURLOPT_RESOLVE, host);
    curl_easy_setopt(curl, CURLOPT_URL, url);

    //curl_easy_setopt(curl, CURLOPT_MAXCONNECTS, 1000L);

    //curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1);
    //curl_easy_setopt(curl, CURLOPT_COOKIEJAR, "cookies.txt");
    //curl_easy_setopt(curl, CURLOPT_COOKIEFILE, "cookies.txt");

    //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    //curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
    //curl_easy_setopt(curl, CURLOPT_DNS_LOCAL_IP4, "127.0.0.1");
    //curl_easy_setopt(curl, CURLOPT_DNS_LOCAL_IP4, "1.1.1.1");
    curl_easy_setopt(curl, CURLOPT_DNS_CACHE_TIMEOUT, 86400L);

    //curl_easy_setopt(curl, CURLOPT_TCP_FASTOPEN, 1L);


    #ifndef RECONNECT
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 120L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL, 60L);
    #endif


    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    //curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
    //curl_easy_setopt(curl, CURLOPT_HAPPY_EYEBALLS_TIMEOUT_MS, 10L);

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_response);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_result);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, command);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, strlen(command));
	curl_easy_setopt(curl, CURLOPT_USERNAME, daemon_username);
	curl_easy_setopt(curl, CURLOPT_PASSWORD, daemon_password);
	//curl_easy_setopt(curl, CURLOPT_USERPWD, userpwd);

    status = curl_easy_perform(curl);
    if(status != 0)
    {
        fprintf(stderr, "error: unable to request data from %s:\n", url);
        fprintf(stderr, "%s\n", curl_easy_strerror(status));
        //logprint(LOG_DEBUG, "error: unable to request data from %s:", url);
        //logprint(LOG_DEBUG, "%s", curl_easy_strerror(status));
        goto error;
    }

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if(code != 200)
    {
        fprintf(stderr, "error: server responded with code %ld\n", code);
        data[write_result.pos] = '\0';
        fprintf(stderr, RED "%s\n" RESET, data);
        //logprint(LOG_DEBUG, "error: server responded with code %ld", code);
        goto error;
    }

    #ifdef RECONNECT
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    curl_global_cleanup();
    #endif

    /* zero-terminate the result */
    data[write_result.pos] = '\0';

    return data;

error:
    #ifdef RECONNECT
    if(data)
        free(data);
    if(curl)
        curl_easy_cleanup(curl);
    if(headers)
        curl_slist_free_all(headers);
    curl_global_cleanup();
    #endif
    return NULL;
}

