export default function retry(fn, retries=3, delay =1000) {
    return fn()
        .then(data => data)
        .catch((err) => {
            if (retries === 0) {
                throw new Error(err);
            }
            console.log("retrying...")
            return new Promise((resolve, reject) => {
                setTimeout(() => {
                    retry(fn, retries -1, delay)
                    .then(resolve)
                    .catch(reject)
                }, delay);
            })
        });
}
