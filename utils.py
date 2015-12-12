# encoding: utf-8

import time
import traceback
from functools import wraps


def debug_exec(*deco_args, **deco_kwargs):
    ''' Выводит в logger.debug время выполнения функции.
    Дополнительне возможности:
    profile = True  - профилировка при помощи cProfile,
    stat_profile = True - профилировка при помощи statprof,
    traceback = True - печатает traceback перед каждым вызовом
    queries = True - выводит запросы, сделанные при выполнении функции
    queries_limit (по умолчанию 50) - лимит при печати запросов
    log_fn - функция для логирования (по умолчанию logger.debug),
    '''
    def deco(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            if deco_kwargs.get('traceback'):
                traceback.print_stack()
            print 'starting %s' % fn.__name__
            start = time.time()
            stat_profile = deco_kwargs.get('stat_profile')
            if stat_profile:
                import statprof
                statprof.reset(frequency=1000)
                statprof.start()
            try:
                return fn(*args, **kwargs)
            finally:
                fn_name = fn.__name__
                print 'finished %s in %.3f s' % (fn_name, time.time() - start)
                if stat_profile:
                    statprof.stop()
                    statprof.display()
        if deco_kwargs.get('profile'):
            import profilehooks
            inner = profilehooks.profile(immediate=True)(inner)
        return inner
    if deco_args:
        return deco(deco_args[0])
    else:
        return deco


def chunks(lst, n):
    for i in xrange(0, len(lst), n):
        yield lst[i:i+n]
