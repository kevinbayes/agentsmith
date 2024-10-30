import * as rxjs from 'rxjs';
import {logger} from "../../utils/logging";

export class BaseBloc {

  analyticsLogger;
  logger;

  constructor(options) {
    this.subject = new rxjs.BehaviorSubject({
      initialised: false,
      ...options,
    });

    this.analyticsLogger = options.analyticsLogger;
    this.logger = options.logger || logger;

    this.events = new rxjs.Subject();
  }

  __makeInitialised = (data) => {
    if(data) {
      this.__updateSubject({initialised: true, ...data});
      this.__publishEvent("INITIALISED", {initialised: true, ...data})
    } else {
      this.__updateSubject({initialised: true});
      this.__publishEvent("INITIALISED", {})
    }
  }

  __updateSubject = (value) =>
      this.subject.next({
        ...this.subject.value,
        ...value,
      });

  __publishEvent = (event, data) => {
    this.events.next({ event, data });
  }

  subscribeToEvents = (func) => this.events.subscribe(func);
  subscribeToState = (func) => this.subject.subscribe(func);

  startWorking = (data) => {
    this.__updateSubject({
        working: true,
        ...data,
    });
  }

  finishedWorking = (data) => {
    this.__updateSubject({
        working: false,
        ...data,
    });
  }

  notWorking = () => {
    this.__updateSubject({
      working: false,
    })
  }

  logAnalyticsEvent(event, data) {
    if (this.analyticsLogger) {
      this.analyticsLogger.log(event, data);
    }
  }
}
