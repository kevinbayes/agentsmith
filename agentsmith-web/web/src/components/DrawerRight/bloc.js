import {BaseBloc} from "../BaseComponent/bloc.js";

export class Bloc extends BaseBloc {

  globalBloc;

  constructor(options, globalContext) {
    super({
      ...options
    });
    this.globalBloc = globalContext.bloc;
  }

  initialise = () => {
    this.globalEventSubscription = this.globalBloc.subscribeToEvents(this.__handleEvent);
  }

  __handleEvent = (event) => {

  }

  close = () => {
    if(this.eventSubscription) {
      this.stateSubscription.unsubscribe();
    }
  }

}

export class Event {

}
