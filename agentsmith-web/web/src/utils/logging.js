export class Logger {
  constructor() {
    this.level = import.meta.env.VITE_APP_LOGGING_LEVEL || 'INFO';
  }

  debug = (...msg) => {
    if (['DEBUG'].includes(this.level)) console.log(...['debug: ', ...msg]);
  };

  info = (...msg) => {
    if (['INFO', 'DEBUG'].includes(this.level)) console.log(...['info: ', ...msg]);
  };

  error = (...msg) => {
    console.error(...msg);
  };
}
export const logger = new Logger();
