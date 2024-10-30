import {BaseBloc} from "../components/BaseComponent/bloc.js";
import {logger} from "../utils/logging.js";
import {AccountStore} from "../model/account.js";

export class Bloc extends BaseBloc {

  accountStore;

  constructor(options) {
    super({ user: options.auth0.user, account: undefined });
    this.auth0 = options.auth0;
    this.accountStore = options?.accountStore || new AccountStore(this.auth0);
  }

  initialise = () => {

    const { user } = this.subject.value;
    logger.info(`initialise bloc for ${user.sub}`);
    this.accountStore.loadProfile()
      .then(response => {
        if(response.status < 300) {
          this.__makeInitialised({ user, profile: response.data });
        } else if(response.status < 404) {
          this.__makeInitialised({ user, });
        } else {
          this.__makeInitialised({ user, error: response.data });
        }
      }, error => {
        logger.error(error);
        this.__makeInitialised({ user, error: error });
      });
  }

  stores = () => {
    return { accountStore: this.accountStore };
  }

  setProfile = (profile) => {
    this.__updateSubject({ profile });
  }

}

export class Event {

}
